"""Run frozen predictors in disposable Verifiers Docker or Prime runtimes.

Only public input files and the callable wrapper enter these runtimes. Hidden
truths and scoring run in the evaluator after prediction files are collected.
"""

import asyncio
import io
import json
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from .artifact_store import archive_info, read_artifact_files, snapshot_artifact
from .sandbox import ExecutionLimitError, ExecutionLimits, SandboxError, SandboxInfrastructureError

PUBLIC_IMAGE = "python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
PACKAGES = ["numpy==2.3.5", "scipy==1.16.2", "scikit-learn==1.7.2", "matplotlib==3.10.7"]
PROCESS_ENV = dict(
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
    MPLCONFIGDIR="/tmp/matplotlib",
)


async def prepare_runtime(runtime):
    """Install the public scientific stack during trusted setup, before egress closes."""
    result = await runtime.run(["python", "-m", "pip", "install", "--disable-pip-version-check", *PACKAGES], {})
    if result.exit_code:
        raise SandboxInfrastructureError("scientific runtime installation failed: " + result.stderr[-2000:])


class RuntimeSandbox:
    def __init__(self, observations, *, artifact, runtime_config=None, limits=None):
        self.limits = limits or ExecutionLimits()
        runtime_config = runtime_config or vf.PrimeConfig(image=PUBLIC_IMAGE, allow=[])
        if runtime_config.type not in ("docker", "prime") or not runtime_config.network_restricted:
            raise ValueError("predictor runtime must be Docker or Prime with allow=[]")
        config = runtime_config.model_copy(
            update=dict(cpu=self.limits.cpus, memory=self.limits.memory_gib, workdir="/workspace", allow=[], block=[])
        )
        self.image = config.image
        self.runner = asyncio.Runner()
        self.context = provision_runtime(config, env=PROCESS_ENV)
        self.runtime = None
        self.closed = False
        try:
            with tempfile.TemporaryDirectory(prefix="physim-predictor-input-") as directory:
                frozen = Path(directory) / "artifact"
                snapshot_artifact(artifact, frozen, self.limits)
                self.runner.run(self.start(frozen, observations))
        except BaseException:
            self.close()
            raise

    async def start(self, artifact, observations):
        self.runtime = await self.context.__aenter__()
        await self.runtime.prepare_setup()
        await prepare_runtime(self.runtime)
        result = await self.runtime.run(["mkdir", "-p", "/workspace", "/observations", "/runtime", "/output"], {})
        if result.exit_code:
            raise SandboxInfrastructureError("could not initialize predictor runtime")
        for name, data in read_artifact_files(artifact, archive_info(artifact)["files"]):
            parent = str((Path("/workspace") / name).parent)
            result = await self.runtime.run(["mkdir", "-p", parent], {})
            if result.exit_code:
                raise SandboxInfrastructureError("could not restore predictor directory")
            await self.runtime.write("/workspace/" + name, data)
        for path in sorted(Path(observations).glob("*.npz")):
            if path.is_symlink() or path.stat().st_size > self.limits.file_mib * 1024 * 1024:
                raise SandboxError("invalid observation file")
            await self.runtime.write("/observations/" + path.name, path.read_bytes())
        await self.runtime.write("/runtime/worker.py", Path(__file__).with_name("container_worker.py").read_bytes())
        result = await self.runtime.run(
            [
                "sh",
                "-c",
                "chmod -R a-w /workspace /observations /runtime; chmod 1777 /output; chmod 755 /workspace /observations /runtime",
            ],
            {},
        )
        if result.exit_code:
            raise SandboxInfrastructureError("could not protect predictor inputs")
        await self.runtime.prepare_execution([])

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.runner.run(self.context.__aexit__(None, None, None))
        finally:
            self.runner.close()

    def prediction(self, actions, queries, *, n_samples, seed, n_ports=12):
        return self.runner.run(self.predict(actions, queries, n_samples, seed, n_ports))

    async def predict(self, actions, queries, n_samples, seed, n_ports):
        request = dict(
            mode="predict", actions=actions, queries=queries, n_samples=n_samples, seed=seed, n_ports=n_ports
        )
        await self.runtime.write("/runtime/request.json", json.dumps(request).encode())
        # Dropping uid protects frozen files even though Runtime.run uses root for setup.
        launcher = """import os,sys
os.setgroups([]); os.setgid(1000); os.setuid(1000)
with open('/runtime/request.json','rb') as f: os.dup2(f.fileno(),0)
os.execv(sys.executable,[sys.executable,'/runtime/worker.py'])
"""
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.limits.wall_seconds + 15):
                result = await self.runtime.run(
                    [
                        "sh",
                        "-c",
                        'rm -f /output/prediction.json; exec timeout -k 2 "$1" python -c "$2" > /output/worker.log 2>&1',
                        "predict",
                        str(self.limits.wall_seconds),
                        launcher,
                    ],
                    dict(
                        PROCESS_ENV,
                        PHYSIM_PREDICTOR_CPU_SECONDS=str(self.limits.cpu_seconds),
                        PHYSIM_PREDICTOR_FILE_MIB=str(self.limits.file_mib),
                    ),
                )
        except TimeoutError as exc:
            raise ExecutionLimitError("predictor exceeded wall-clock limit") from exc
        log_result = await self.runtime.run(["tail", "-c", "12000", "/output/worker.log"], {})
        execution = dict(
            exit_code=result.exit_code,
            output=log_result.stdout,
            wall_seconds=time.monotonic() - started,
            execution_limits=asdict(self.limits),
        )
        if result.exit_code:
            error = ExecutionLimitError if result.exit_code in (124, 137, 152, 153) else SandboxError
            raise error(f"predictor exited with status {result.exit_code}: {log_result.stdout[-3000:]}")
        result = await self.runtime.run(["tar", "-C", "/output", "-cf", "/runtime/result.tar", "prediction.json"], {})
        if result.exit_code:
            raise SandboxError("predictor produced no prediction file")
        raw = await self.runtime.read("/runtime/result.tar", max_bytes=(self.limits.file_mib + 1) * 1024 * 1024)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            entries = archive.getmembers()
            if len(entries) != 1 or not entries[0].isfile() or entries[0].size > self.limits.file_mib * 1024 * 1024:
                raise SandboxError("prediction must be a bounded regular JSON file")
            data = archive.extractfile(entries[0]).read()
        return data, execution
