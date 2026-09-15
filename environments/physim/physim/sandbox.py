"""Docker boundary for analysis and prediction; no simulator mounts or network."""

from __future__ import annotations

import hashlib
import io
import json
import selectors
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
IMAGE = "physim-predictor:0.12.0"
OUTPUT_LIMIT = 128 * 1024
ARTIFACT_LIMIT = 64 * 1024 * 1024


class SandboxError(RuntimeError):
    pass


def docker(arguments, *, data=None, timeout=30):
    result = subprocess.run(
        ["docker", *arguments], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    if result.returncode:
        raise SandboxError(result.stderr.decode(errors="replace")[:2000])
    return result.stdout


class Sandbox:
    def __init__(self, observations, *, artifact=None, image=IMAGE):
        self.name = "r6-pilot-" + uuid.uuid4().hex[:12]
        self.closed = False
        self.image = image
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
            "1",
            "--memory",
            "1g",
            "--memory-swap",
            "1g",
            "--user",
            "1000:1000",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
            "--tmpfs",
            "/output:rw,nosuid,nodev,size=64m,mode=1777",
            "--mount",
            f"type=bind,source={runtime},target=/runtime,readonly",
            "--mount",
            f"type=bind,source={self.public_observations},target=/observations,readonly",
        ]
        if artifact is None:
            args += ["--tmpfs", "/workspace:rw,nosuid,nodev,size=128m,mode=1777"]
        else:
            artifact = Path(artifact).resolve()
            staged = self.public_root / "artifact"
            staged.mkdir()
            total = 0
            for source in artifact.rglob("*"):
                if source.is_symlink():
                    raise SandboxError("artifact symlinks are forbidden")
                if source.is_file():
                    total += source.stat().st_size
                    if total > ARTIFACT_LIMIT:
                        raise SandboxError("artifact size exceeds cap")
                    dest = staged / source.relative_to(artifact)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
            args += ["--mount", f"type=bind,source={staged},target=/workspace,readonly"]
        args += ["--workdir", "/workspace", image, "sleep", "infinity"]
        try:
            self.container_id = docker(args).decode().strip()
        except BaseException:
            shutil.rmtree(self.public_root)
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
                    raise SandboxError("execution exceeded wall-clock limit; analysis session terminated")
                for key, _ in selector.select(timeout=0.2):
                    part = key.fileobj.read1(8192)
                    if not part:
                        selector.unregister(key.fileobj)
                        continue
                    captured.extend(part)
                    if len(captured) > output_limit:
                        raise SandboxError("execution exceeded output limit; analysis session terminated")
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
        """Extract only bounded regular files; no symlinks, devices or traversal."""
        target = Path(target)
        if target.exists():
            raise SandboxError("artifact snapshot directory already exists")
        target.mkdir(parents=True)
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
        total, files = 0, []
        try:
            with tarfile.open(fileobj=stream.stdout, mode="r|") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise SandboxError("unsafe artifact member path")
                    if member.isdir():
                        continue
                    if not member.isfile() or member.issym() or member.islnk():
                        raise SandboxError("artifact may contain only regular files and directories")
                    total += member.size
                    if member.size > 20 * 1024 * 1024 or total > ARTIFACT_LIMIT or len(files) >= 2000:
                        raise SandboxError("artifact exceeds file or total-size limit")
                    dest = target.joinpath(*path.parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, dest.open("xb") as output:
                        while part := source.read(65536):
                            output.write(part)
                    files.append(
                        dict(
                            path=dest.relative_to(target).as_posix(),
                            bytes=member.size,
                            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        )
                    )
            if stream.wait(timeout=10):
                raise SandboxError("artifact export failed")
        except BaseException:
            stream.kill()
            stream.wait()
            raise
        finally:
            stream.stdout.close()
            stream.stderr.close()
        if not (target / "predictor.py").is_file():
            raise SandboxError("write /workspace/predictor.py before submitting")
        return dict(files=files, bytes=total)

    def prediction(self, actions, queries, *, n_samples, seed, n_ports=12):
        # Remove any result from a preceding prediction in this container.
        docker(["exec", self.name, "rm", "-f", "/output/prediction.json"])
        result = self.invoke(
            dict(mode="predict", actions=actions, queries=queries, n_samples=n_samples, seed=seed, n_ports=n_ports)
        )
        if result["exit_code"]:
            raise SandboxError(result["output"][-3000:])
        # JSON artifact is capped by the in-container per-file limit. Tar does
        # not follow symlinks by default; reject every nonregular member.
        raw = docker(["exec", self.name, "tar", "-C", "/output", "-cf", "-", "prediction.json"])
        if len(raw) > 21 * 1024 * 1024:
            raise SandboxError("prediction exceeds data-only transfer cap")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            entries = archive.getmembers()
            if len(entries) != 1 or not entries[0].isfile() or entries[0].size > 20 * 1024 * 1024:
                raise SandboxError("prediction must be a bounded regular JSON file")
            data = archive.extractfile(entries[0]).read()
        return data, result
