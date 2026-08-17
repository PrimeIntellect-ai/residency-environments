"""Process boundary for user-authored Python strategy callbacks."""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alphaverse.strategy.policy import validate_strategy_source
from alphaverse.strategy.protocol import ActionBatch, InputEnvelope
from alphaverse.strategy.sdk import Strategy


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    version_id: str
    entrypoint: str
    source: str

    @classmethod
    def build(cls, source: str, *, entrypoint: str = "strategy:StrategyImpl"):
        return cls._build(source, entrypoint=entrypoint, trusted=False)

    @classmethod
    def build_trusted(cls, source: str, *, entrypoint: str = "strategy:StrategyImpl"):
        """Build evaluator-owned source; never expose this through an agent API."""

        return cls._build(source, entrypoint=entrypoint, trusted=True)

    @classmethod
    def _build(cls, source: str, *, entrypoint: str, trusted: bool):
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if not source.strip():
            raise ValueError("source must not be empty")
        if len(source.encode()) > 1_000_000:
            raise ValueError("source exceeds the 1 MB MVP artifact limit")
        if not trusted:
            validate_strategy_source(source)
        module, separator, class_name = entrypoint.partition(":")
        if not separator or module != "strategy" or not class_name:
            raise ValueError("entrypoint must have the form strategy:ClassName")
        digest = hashlib.sha256((entrypoint + "\0" + source).encode()).hexdigest()
        return cls(f"sha256:{digest}", entrypoint, source)


class SubprocessStrategy(Strategy):
    """Strategy proxy speaking canonical envelopes to a dedicated child process.

    This is a process boundary and resource backstop, not a complete hostile-code
    sandbox. Production evaluation should additionally place it in its own restricted
    container or Prime sandbox.
    """

    def __init__(
        self,
        artifact: StrategyArtifact,
        *,
        participant_id: str,
        strategy_instance_id: str,
        product_id: str,
        parameters: dict[str, object] | None = None,
        seed: int = 0,
        max_actions_per_callback: int = 100,
        callback_timeout_ns: int = 1_000_000_000,
        memory_limit_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if not isinstance(artifact, StrategyArtifact):
            raise TypeError("artifact must be a StrategyArtifact")
        if callback_timeout_ns <= 0:
            raise ValueError("callback_timeout_ns must be positive")
        if memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        self.artifact = artifact
        self.callback_timeout = callback_timeout_ns / 1_000_000_000
        self._directory = tempfile.TemporaryDirectory(prefix="alphaverse-strategy-")
        source_path = Path(self._directory.name) / "strategy.py"
        source_path.write_text(artifact.source, encoding="utf-8")
        package_root = str(Path(__file__).resolve().parents[2])
        bootstrap = (
            "import runpy,sys;"
            "sys.path.insert(0,sys.argv.pop(1));"
            "runpy.run_module('alphaverse.strategy.worker',run_name='__main__')"
        )
        argv = [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            package_root,
            str(source_path),
            artifact.entrypoint,
            participant_id,
            strategy_instance_id,
            product_id,
            json.dumps(parameters or {}, separators=(",", ":"), sort_keys=True),
            str(seed),
            str(max_actions_per_callback),
        ]
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            cwd=self._directory.name,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
        )
        if sys.platform.startswith("linux"):
            import resource

            if hasattr(resource, "prlimit"):
                resource.prlimit(
                    self._process.pid,
                    resource.RLIMIT_AS,
                    (memory_limit_bytes, memory_limit_bytes),
                )
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], self.callback_timeout)
        if not ready:
            self.close()
            raise TimeoutError("strategy process did not initialize before its deadline")
        line = self._process.stdout.readline()
        if not line:
            detail = self._failure("strategy process failed to initialize")
            self.close()
            raise RuntimeError(detail)
        if json.loads(line) != {"ready": True}:
            self.close()
            raise RuntimeError("strategy process returned an invalid initialization message")

    def _handle(self, event: InputEnvelope) -> ActionBatch:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(self._failure("strategy process exited"))
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(event.to_json() + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], self.callback_timeout)
        if not ready:
            self.close()
            raise TimeoutError("strategy callback exceeded its wall-time limit")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(self._failure("strategy process closed stdout"))
        decoded = json.loads(line)
        if "error" in decoded:
            raise RuntimeError(f"strategy callback failed: {decoded['error']}: {decoded.get('message', '')}")
        return ActionBatch.from_json(line)

    def _failure(self, prefix: str) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        detail = self._stderr.read()[-4_000:].strip()
        return f"{prefix}: {detail}" if detail else prefix

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        stderr = getattr(self, "_stderr", None)
        if stderr is not None and not stderr.closed:
            stderr.close()
        directory = getattr(self, "_directory", None)
        if directory is not None:
            directory.cleanup()

    def on_start(self, ctx, event):
        return self._handle(event)

    def on_market(self, ctx, event):
        return self._handle(event)

    def on_levels(self, ctx, event):
        return self._handle(event)

    def on_execution(self, ctx, event):
        return self._handle(event)

    def on_timer(self, ctx, event):
        return self._handle(event)

    def on_risk(self, ctx, event):
        return self._handle(event)

    def on_stop(self, ctx, event):
        return self._handle(event)

    def on_signal(self, ctx, event):
        return self._handle(event)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["StrategyArtifact", "SubprocessStrategy"]
