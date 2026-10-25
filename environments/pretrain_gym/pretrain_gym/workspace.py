"""Files and commands in a sandbox reached by id from a host process: a Docker container or a Prime VM.

The agent's workspace is one; a Prime scorer VM is another. Archives built inside a sandbox may be
crafted by the agent, so they are unpacked with `tarfile`'s `data` filter, which refuses absolute
paths, `..`, links leaving the destination, and device files.
"""

import asyncio
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Literal

import zstandard

WORKDIR = "/workspace"
# The Prime gateway buffers whole files, so large transfers go in parts.
PART_BYTES = 256 * 2**20
TRANSFER_TIMEOUT = 1800
SandboxKind = Literal["docker", "prime"]


class WorkspaceError(Exception):
    """A path the agent controls is missing or unusable."""


def untar(stream: BinaryIO, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        archive.extractall(destination, filter="data")


class DockerBox:
    def __init__(self, container: str, root: str = WORKDIR) -> None:
        self.container, self.root = container, root

    async def _docker(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(f"docker {shlex.join(args[:3])} failed: {stderr.decode(errors='replace').strip()}")
        return stdout.decode()

    async def sh(self, command: str) -> str:
        return await self._docker("exec", "--workdir", self.root, self.container, "sh", "-c", command)

    async def exists(self, path: str) -> bool:
        return (await self.sh(f"[ -e {shlex.quote(path)} ] && echo yes || true")).strip() == "yes"

    async def get_tree(self, path: str, destination: Path) -> None:
        """Copy `path` into the host directory `destination`, replacing symlinks with their targets."""
        argv = ["docker", "exec", self.container, "tar", "-C", self.root, "-chf", "-", path]

        def copy() -> None:
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                untar(process.stdout, destination)
                stderr = process.stderr.read()
            if process.returncode:
                raise RuntimeError(f"copying {path} out failed: {stderr.decode(errors='replace').strip()}")

        await asyncio.to_thread(copy)

    async def put_tree(self, source: Path, path: str) -> None:
        """Copy the contents of the host directory `source` to `path`."""
        target = f"{self.root}/{path}"
        await self._docker("exec", self.container, "mkdir", "-p", target)
        await self._docker("cp", f"{source}/.", f"{self.container}:{target}")


class PrimeBox:
    def __init__(self, sandbox_id: str, client, root: str = WORKDIR) -> None:
        self.id, self.client, self.root = sandbox_id, client, root

    def _abs(self, path: str) -> str:
        return f"{self.root.rstrip('/')}/{path}"

    async def sh(self, command: str) -> str:
        result = await self.client.execute_command(
            self.id, f"bash -o pipefail -c {shlex.quote(command)}", working_dir=self.root, timeout=TRANSFER_TIMEOUT
        )
        if result.exit_code:
            raise RuntimeError(f"`{command[:80]}` failed in sandbox {self.id}: {(result.stderr or '').strip()[-2000:]}")
        return result.stdout or ""

    async def exists(self, path: str) -> bool:
        return (await self.sh(f"[ -e {shlex.quote(path)} ] && echo yes || true")).strip() == "yes"

    async def put_file(self, local: Path, remote: str) -> None:
        """Upload the host file `local` to the absolute path `remote`."""
        with local.open("rb") as file:
            for index in range(max(1, -(-local.stat().st_size // PART_BYTES))):
                chunk = await asyncio.to_thread(file.read, PART_BYTES)
                part = f"{remote}.part{index:05d}"
                await self.client.upload_bytes(self.id, part, chunk, filename=Path(part).name, timeout=TRANSFER_TIMEOUT)
        quoted = shlex.quote(remote)
        await self.sh(f"cat {quoted}.part* > {quoted} && rm -f {quoted}.part*")

    async def get_file(self, remote: str, local: Path) -> None:
        """Download the absolute path `remote` to the host file `local`."""
        quoted = shlex.quote(remote)
        await self.sh(f"split -b {PART_BYTES} -d -a 5 {quoted} {quoted}.part")
        parts = sorted((await self.sh(f"ls {quoted}.part* 2>/dev/null || true")).split())
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with local.open("wb") as output:
                for part in parts:
                    with tempfile.NamedTemporaryFile(dir=local.parent) as piece:
                        await self.client.download_file(self.id, part, piece.name, timeout=TRANSFER_TIMEOUT)
                        await asyncio.to_thread(output.write, Path(piece.name).read_bytes())
        finally:
            await self.sh(f"rm -f {quoted}.part*")

    async def get_tree(self, path: str, destination: Path) -> None:
        """Copy `path` into the host directory `destination`, replacing symlinks with their targets."""
        archive = f"/var/tmp/pretrain-gym-{uuid.uuid4().hex}.tar.zst"
        await self.sh(f"tar -chf - {shlex.quote(path)} | zstd -q -1 -T0 -o {archive}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
                local = Path(scratch) / "tree.tar.zst"
                await self.get_file(archive, local)

                def unpack() -> None:
                    with local.open("rb") as file, zstandard.ZstdDecompressor().stream_reader(file) as stream:
                        untar(stream, destination)

                await asyncio.to_thread(unpack)
        finally:
            await self.sh(f"rm -f {archive}")

    async def put_tree(self, source: Path, path: str) -> None:
        """Copy the contents of the host directory `source` to `path`."""
        archive = f"/var/tmp/pretrain-gym-{uuid.uuid4().hex}.tar.zst"
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "tree.tar.zst"

            def pack() -> None:
                with local.open("wb") as file, zstandard.ZstdCompressor(level=1, threads=-1).stream_writer(file) as out:
                    with tarfile.open(fileobj=out, mode="w|") as tar:
                        tar.add(source, arcname=".")

            await asyncio.to_thread(pack)
            await self.put_file(local, archive)
        target = shlex.quote(path)
        await self.sh(
            f"mkdir -p {target} && zstd -dcq {archive} | tar -xf - -C {target}; s=$?; rm -f {archive}; exit $s"
        )


def open_box(kind: SandboxKind, sandbox_id: str, client=None, root: str = WORKDIR) -> DockerBox | PrimeBox:
    return DockerBox(sandbox_id, root) if kind == "docker" else PrimeBox(sandbox_id, client, root)
