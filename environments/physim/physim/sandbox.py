"""Docker boundary for analysis and prediction; no simulator mounts or network."""

from __future__ import annotations

import io
import json
import selectors
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifact_store import ARCHIVE, MARKER, SandboxError, archive_workspace, snapshot_artifact

HERE = Path(__file__).resolve().parent
IMAGE = "physim-predictor:0.12.0"
OUTPUT_LIMIT = 128 * 1024
ARTIFACT_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class ExecutionLimits:
    """Host-owned resource policy, shared by validation and grading."""

    cpu_seconds: int = 20
    wall_seconds: int = 30
    cpus: int = 1
    memory_gib: int = 1
    artifact_mib: int = 64
    file_mib: int = 20
    temporary_mib: int = 128

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("predictor execution limits must be positive integers")


class SandboxInfrastructureError(SandboxError):
    """Host/container transport failure; details are not predictor feedback."""


class ExecutionLimitError(SandboxError):
    """An execution resource boundary interrupted the submitted program."""


def docker(arguments, *, data=None, timeout=30):
    result = subprocess.run(
        ["docker", *arguments], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace")[:2000]
        raise SandboxInfrastructureError(
            f"docker {arguments[0]} failed (exit code {result.returncode}): {detail or 'no diagnostic output'}"
        )
    return result.stdout


class Sandbox:
    def __init__(self, observations, *, artifact=None, image=IMAGE, limits=None):
        self.name = "r6-pilot-" + uuid.uuid4().hex[:12]
        self.closed = False
        self.workspace_volume = None
        self.image = image
        self.limits = limits or ExecutionLimits()
        observations = Path(observations).resolve()
        observations.mkdir(parents=True, exist_ok=True)
        # Stage only public runtime/data in a portable temporary directory.
        self.public_root = Path(tempfile.mkdtemp(prefix="physim-r6-public-"))
        runtime = self.public_root / "runtime"
        runtime.mkdir()
        (runtime / "container_worker.py").write_bytes((HERE / "container_worker.py").read_bytes())
        self.public_observations = self.public_root / "observations"
        self.public_observations.mkdir()
        self.sync_observations(observations)
        args = [
            "run",
            "-d",
            "--name",
            self.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--cpus",
            str(self.limits.cpus),
            "--memory",
            f"{self.limits.memory_gib}g",
            "--memory-swap",
            f"{self.limits.memory_gib}g",
            "--env",
            f"PHYSIM_PREDICTOR_CPU_SECONDS={self.limits.cpu_seconds}",
            "--env",
            f"PHYSIM_PREDICTOR_FILE_MIB={self.limits.file_mib}",
            "--user",
            "1000:1000",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.limits.temporary_mib}m,mode=1777",
            "--tmpfs",
            f"/output:rw,nosuid,nodev,size={max(64, self.limits.file_mib * 2)}m,mode=1777",
            "--mount",
            f"type=bind,source={runtime},target=/runtime,readonly",
            "--mount",
            f"type=bind,source={self.public_observations},target=/observations,readonly",
        ]
        try:
            if artifact is None:
                args += ["--tmpfs", "/workspace:rw,nosuid,nodev,size=128m,mode=1777"]
            else:
                staged = self.public_root / "artifact"
                snapshot_artifact(artifact, staged, self.limits)
                self.workspace_volume = self.name + "-workspace"
                docker(["volume", "create", self.workspace_volume])
                # Restore in Linux so case/Unicode semantics never depend on
                # the host's filesystem. This helper executes only trusted tar.
                docker(
                    [
                        "run",
                        "--rm",
                        "--name",
                        self.name + "-restore",
                        "--network",
                        "none",
                        "--read-only",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        "64",
                        "--memory",
                        # File-backed reads and writes count toward the helper's
                        # cgroup memory. Allow both copies of the permitted
                        # archive plus tar overhead, independently of prediction.
                        f"{256 + 2 * self.limits.artifact_mib}m",
                        "--cpus",
                        "1",
                        "--user",
                        "0:0",
                        "--mount",
                        f"type=bind,source={staged},target=/snapshot,readonly",
                        "--mount",
                        f"type=volume,source={self.workspace_volume},target=/workspace,volume-nocopy",
                        image,
                        "tar",
                        "--no-same-owner",
                        "-xf",
                        f"/snapshot/{ARCHIVE}",
                        "-C",
                        "/workspace",
                    ],
                    timeout=120,
                )
                args += [
                    "--mount",
                    f"type=volume,source={self.workspace_volume},target=/workspace,readonly,volume-nocopy",
                ]
            args += ["--workdir", "/workspace", image, "sleep", "infinity"]
            self.container_id = docker(args).decode().strip()
        except BaseException:
            # Also cover helper timeouts and worker startup failures. Cleanup
            # commands must not mask the original exception.
            cleanup = [["docker", "rm", "-f", name] for name in (self.name + "-restore", self.name)]
            if self.workspace_volume:
                cleanup.append(["docker", "volume", "rm", self.workspace_volume])
            for command in cleanup:
                try:
                    subprocess.run(command, capture_output=True, timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    pass  # Preserve the original startup error if Docker is unavailable.
            shutil.rmtree(self.public_root, ignore_errors=True)
            raise

    def sync_observations(self, directory):
        for source in Path(directory).glob("*.npz"):
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 20 * 1024 * 1024:
                raise SandboxError("observation must be a bounded regular NPZ file")
            target = self.public_observations / source.name
            if not target.exists():
                shutil.copyfile(source, target)

    def close(self):
        if not self.closed:
            docker(["rm", "-f", self.name])
            self.closed = True
            try:
                if self.workspace_volume:
                    docker(["volume", "rm", self.workspace_volume])
            finally:
                shutil.rmtree(self.public_root)

    def invoke(self, request, *, timeout=30, output_limit=OUTPUT_LIMIT):
        encoded = json.dumps(request).encode()
        if len(encoded) > 256_000:
            raise SandboxError("tool request exceeds byte limit")
        process = subprocess.Popen(
            ["docker", "exec", "-i", self.name, "python", "/runtime/container_worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            process.stdin.write(encoded)
            process.stdin.close()
        except BrokenPipeError:
            pass
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        captured = bytearray()
        started = time.monotonic()
        try:
            while selector.get_map():
                if time.monotonic() - started > timeout:
                    raise ExecutionLimitError("execution exceeded wall-clock limit; analysis session terminated")
                for key, _ in selector.select(timeout=0.2):
                    part = key.fileobj.read1(8192)
                    if not part:
                        selector.unregister(key.fileobj)
                        continue
                    captured.extend(part)
                    if len(captured) > output_limit:
                        raise ExecutionLimitError("execution exceeded output limit; analysis session terminated")
            status = process.wait(timeout=2)
        except BaseException:
            self.close()  # Also kills descendants; terminating docker exec alone does not.
            process.kill()
            process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()
        text = captured.decode(errors="replace")
        return dict(
            exit_code=status, output=text[-12000:], output_bytes=len(captured), wall_seconds=time.monotonic() - started
        )

    def python(self, code, *, timeout=30):
        if type(code) is not str or len(code) > 24_000:
            raise SandboxError("python code must be a string of at most 24000 characters")
        return self.invoke(dict(mode="python", code=code), timeout=timeout)

    def export_workspace(self, target, *, excludes=()):
        """Archive bounded regular files without host filename translation."""
        limits = getattr(self, "limits", ExecutionLimits())
        target = Path(target)
        if target.exists():
            raise SandboxError("artifact snapshot directory already exists")
        # docker cp cannot read tmpfs mounts; stream an archive from inside.
        stream = subprocess.Popen(
            [
                "docker",
                "exec",
                self.name,
                "tar",
                "-C",
                "/workspace",
                *[f"--exclude={pattern}" for pattern in excludes],
                "-cf",
                "-",
                ".",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            snapshot = archive_workspace(stream.stdout, target, limits)
            if stream.wait(timeout=10):
                (target / MARKER).unlink(missing_ok=True)
                raise SandboxError("artifact export failed")
            return snapshot
        except BaseException:
            stream.kill()
            stream.wait()
            raise
        finally:
            stream.stdout.close()
            stream.stderr.close()

    def prediction(self, actions, queries, *, n_samples, seed, n_ports=12):
        # Remove any result from a preceding prediction in this container.
        docker(["exec", self.name, "rm", "-f", "/output/prediction.json"])
        result = self.invoke(
            dict(mode="predict", actions=actions, queries=queries, n_samples=n_samples, seed=seed, n_ports=n_ports),
            timeout=self.limits.wall_seconds,
        )
        result["execution_limits"] = asdict(self.limits)
        if result["exit_code"]:
            detail = result["output"][-3000:]
            if result["exit_code"] in (137, 152, 153) or any(
                marker in detail
                for marker in ("MemoryError", "Cannot allocate memory", "File too large", "No space left on device")
            ):
                raise ExecutionLimitError(
                    f"predictor exited with status {result['exit_code']}; possible resource limit: {detail}"
                )
            raise SandboxError(f"predictor exited with status {result['exit_code']}: {detail}")
        # JSON artifact is capped by the in-container per-file limit. Tar does
        # not follow symlinks by default; reject every nonregular member.
        raw = docker(["exec", self.name, "tar", "-C", "/output", "-cf", "-", "prediction.json"])
        if len(raw) > (self.limits.file_mib + 1) * 1024 * 1024:
            raise SandboxError("prediction exceeds data-only transfer cap")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            entries = archive.getmembers()
            if len(entries) != 1 or not entries[0].isfile() or entries[0].size > self.limits.file_mib * 1024 * 1024:
                raise SandboxError("prediction must be a bounded regular JSON file")
            data = archive.extractfile(entries[0]).read()
        return data, result
