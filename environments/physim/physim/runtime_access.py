"""Private evaluator-side file transport to the framework-owned agent runtime.

The MCP server is a separate trusted process. A local Unix socket lets it ask
the task controller to copy observations or snapshot the workspace through the
public Verifiers Runtime API. This socket is never exposed to the agent.
"""

import asyncio
import io
import json
import secrets
import socket
import tempfile
import uuid
import weakref
from pathlib import Path

from .artifact_store import SandboxError, archive_workspace
from .sandbox import SandboxInfrastructureError


class RuntimeAccess:
    def __init__(self, runtime, output, limits):
        self._runtime = weakref.ref(runtime)
        self.output = Path(output)
        self.limits = limits
        self.directory = tempfile.TemporaryDirectory(prefix="physim-io-")
        self.path = str(Path(self.directory.name) / "io.sock")
        self.secret = secrets.token_hex(32)
        self.server = None

    @property
    def runtime(self):
        runtime = self._runtime()
        if runtime is None:
            raise SandboxInfrastructureError("agent runtime has closed")
        return runtime

    async def start(self):
        self.server = await asyncio.start_unix_server(self.handle, self.path, limit=8192)
        weakref.finalize(self.runtime, self.server.close)
        weakref.finalize(self.runtime, self.directory.cleanup)
        return dict(path=self.path, secret=self.secret)

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        self.directory.cleanup()

    async def handle(self, reader, writer):
        try:
            async with asyncio.timeout(180):
                request = json.loads(await reader.readline())
                if not secrets.compare_digest(request.pop("secret", ""), self.secret):
                    raise SandboxInfrastructureError("unauthorized runtime transport")
                name = request["name"]
                if Path(name).name != name or name in (".", ".."):
                    raise SandboxError("invalid transport filename")
                if request["op"] == "snapshot":
                    value = await self.snapshot(self.output / name)
                elif request["op"] == "observation":
                    await self.observation(self.output / "observations" / name)
                    value = None
                else:
                    raise SandboxError("unknown runtime transport operation")
                response = dict(value=value)
        except SandboxError as exc:
            response = dict(error=str(exc), infrastructure=isinstance(exc, SandboxInfrastructureError))
        except Exception as exc:
            response = dict(error=f"{type(exc).__name__}: {exc}", infrastructure=True)
        try:
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def snapshot(self, target):
        remote = "/tmp/physim-workspace-" + uuid.uuid4().hex + ".tar"
        try:
            result = await self.runtime.run(
                [
                    "sh",
                    "-c",
                    'ulimit -f "$1"; exec timeout 60 tar -C /workspace --exclude=./.vf-* -cf "$2" .',
                    "snapshot",
                    str((self.limits.artifact_mib + 12) * 2048),
                    remote,
                ],
                {},
            )
            if result.exit_code:
                raise SandboxError("workspace archive failed or exceeded its transfer limit")
            raw = await self.runtime.read(remote, max_bytes=(self.limits.artifact_mib + 12) * 1024 * 1024)
            return archive_workspace(io.BytesIO(raw), target, self.limits)
        finally:
            await self.runtime.run(["rm", "-f", remote], {})

    async def observation(self, path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > self.limits.file_mib * 1024 * 1024:
            raise SandboxError("observation must be a bounded regular file")
        remote = "/tmp/physim-observation-" + uuid.uuid4().hex
        try:
            await self.runtime.write(remote, path.read_bytes())
            script = """import os,sys
fd=os.open('/observations/'+sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
with os.fdopen(fd,'wb') as dst, open(sys.argv[1],'rb') as src: dst.write(src.read())
"""
            result = await self.runtime.run(["python", "-c", script, remote, path.name], {})
            if result.exit_code:
                raise SandboxInfrastructureError("could not deliver observation to agent runtime")
        finally:
            await self.runtime.run(["rm", "-f", remote], {})


def request_access(state, operation, name):
    access = state.runtime_access
    if not access:
        raise SandboxInfrastructureError("missing trusted runtime transport")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(185)
        connection.connect(access["path"])
        connection.sendall(json.dumps(dict(secret=access["secret"], op=operation, name=name)).encode() + b"\n")
        with connection.makefile("rb") as stream:
            raw = stream.readline(1024 * 1024)
    if not raw.endswith(b"\n"):
        raise SandboxInfrastructureError("invalid runtime transport response")
    result = json.loads(raw)
    if "error" in result:
        error = SandboxInfrastructureError if result["infrastructure"] else SandboxError
        raise error(result["error"])
    return result["value"]
