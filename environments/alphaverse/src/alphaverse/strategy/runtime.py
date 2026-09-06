"""Synchronous strategy callbacks over Verifiers-owned isolated runtimes."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, RuntimeConfig, provision_runtime

from alphaverse.strategy.bundle import public_strategy_files
from alphaverse.strategy.errors import StrategyInfrastructureError
from alphaverse.strategy.protocol import ActionBatch, InputEnvelope
from alphaverse.strategy.sdk import Strategy

MAX_RESPONSE_BYTES = 1_048_576
MAX_STDERR_BYTES = 1_048_576


def validate_trading_runtime(config: RuntimeConfig) -> None:
    """Reject placement that would run uploaded code on the evaluator."""
    if not isinstance(config, (DockerConfig, PrimeConfig)):
        raise ValueError("trading programs require a Docker or Prime runtime")
    if isinstance(config, PrimeConfig) and not config.vm:
        raise ValueError("Prime trading runtimes require vm=true")
    if config.allow or config.block != ["*"]:
        raise ValueError("trading runtimes require allow=[] and no outbound network")


class RuntimeStrategy(Strategy):
    """One persistent program in a fresh, private trading sandbox per deployment."""

    def __init__(
        self,
        artifact,
        *,
        participant_id: str,
        strategy_instance_id: str,
        product_id: str,
        parameters: dict[str, object] | None = None,
        seed: int = 0,
        max_actions_per_callback: int = 100,
        callback_timeout_ns: int = 1_000_000_000,
        memory_limit_bytes: int = 256 * 1024 * 1024,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        if callback_timeout_ns <= 0 or memory_limit_bytes <= 0 or max_actions_per_callback <= 0:
            raise ValueError("strategy resource limits must be positive")
        config = runtime_config or DockerConfig(allow=[])
        validate_trading_runtime(config)
        self.config = config.model_copy(
            update={
                "cpu": config.cpu or 1,
                "memory": config.memory or 1,
                "workdir": "/trading",
            }
        )
        self.artifact = artifact
        self.callback_timeout = callback_timeout_ns / 1_000_000_000
        self.max_actions = max_actions_per_callback
        self._closed = False
        self._buffer = b""
        self._stderr_tail = b""
        self._stderr_failure: Exception | None = None
        self._request_number = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="alphaverse-trading-runtime", daemon=True)
        self._thread.start()
        self._ready.wait()
        arguments = [
            "/trading/strategy.py",
            artifact.entrypoint,
            participant_id,
            strategy_instance_id,
            product_id,
            json.dumps(parameters or {}),
            str(seed),
            str(max_actions_per_callback),
            str(memory_limit_bytes),
        ]
        try:
            self._submit(self._start(arguments))
        except BaseException:
            self.close()
            raise

    def _serve(self) -> None:
        with asyncio.Runner() as runner:
            runner.run(self._lifetime())

    async def _lifetime(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._stack = AsyncExitStack()
        self._ready.set()
        await self._stop.wait()

    def _submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("trading runtime is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            # Preserve event -> action ordering: each remote callback round trip
            # blocks the simulation here, even when its action batch is empty.
            return future.result()
        except SandboxError as exc:
            raise StrategyInfrastructureError(f"trading runtime provider failed: {type(exc).__name__}") from exc

    async def _start(self, arguments: list[str]) -> None:
        try:
            async with asyncio.timeout(900):
                self._runtime = await self._stack.enter_async_context(provision_runtime(self.config))
                await self._runtime.prepare_setup()
                for path, contents in public_strategy_files().items():
                    await self._runtime.write(f"/trading/{path}", contents)
                await self._runtime.write("/trading/strategy.py", self.artifact.source.encode())
                await self._runtime.prepare_execution([])
                # Isolated mode deliberately excludes cwd; only this public SDK root is added.
                bootstrap = (
                    "import runpy,sys;sys.path.insert(0,'/trading');"
                    "runpy.run_module('alphaverse.strategy.trading_worker',run_name='__main__')"
                )
                self._process = await self._runtime.open_process(
                    ["python", "-I", "-u", "-c", bootstrap, *arguments], {}
                )
                self._stdout = self._process.stdout.__aiter__()
                self._stderr_task = asyncio.create_task(self._drain_stderr())
        except Exception as exc:
            raise StrategyInfrastructureError(f"trading runtime setup failed: {type(exc).__name__}") from exc
        try:
            async with asyncio.timeout(self.callback_timeout):
                ready = await self._read_message()
        except TimeoutError as exc:
            raise RuntimeError("strategy initialization exceeded its callback deadline") from exc
        if ready != {"ready": True}:
            raise ValueError("strategy returned an invalid initialization message")

    async def _drain_stderr(self) -> None:
        total = 0
        try:
            async for chunk in self._process.stderr:
                total += len(chunk)
                self._stderr_tail = (self._stderr_tail + chunk)[-4_000:]
                if total > MAX_STDERR_BYTES:
                    self._stderr_failure = ValueError("strategy exceeded its diagnostic-output limit")
                    return
        except Exception as exc:
            self._stderr_failure = StrategyInfrastructureError(
                f"trading diagnostic transport failed: {type(exc).__name__}"
            )

    async def _read_message(self) -> dict:
        while b"\n" not in self._buffer:
            try:
                chunk = await anext(self._stdout)
            except StopAsyncIteration as exc:
                detail = self._stderr_tail.decode(errors="replace")
                raise RuntimeError(f"strategy process exited: {detail}") from exc
            except Exception as exc:
                raise StrategyInfrastructureError(f"trading transport failed: {type(exc).__name__}") from exc
            if len(self._buffer) + len(chunk) > MAX_RESPONSE_BYTES:
                raise ValueError("strategy response exceeded the byte limit")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        if self._stderr_failure is not None:
            raise self._stderr_failure
        decoded = json.loads(line)
        if not isinstance(decoded, dict):
            raise ValueError("strategy response must be an object")
        return decoded

    async def _round_trip(self, payload: dict) -> dict:
        self._request_number += 1
        request_id = self._request_number
        message = {"request_id": request_id, **payload}
        async with asyncio.timeout(self.callback_timeout):
            try:
                await self._process.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise RuntimeError("strategy process closed its input") from exc
            except Exception as exc:
                raise StrategyInfrastructureError(f"trading input transport failed: {type(exc).__name__}") from exc
            response = await self._read_message()
        if type(response.get("request_id")) is not int or response["request_id"] != request_id:
            raise ValueError("strategy response belongs to another callback")
        return response

    async def _ping(self) -> None:
        response = await self._round_trip({"operation": "ping"})
        if set(response) != {"request_id", "ready"} or response["ready"] is not True:
            raise ValueError("staged strategy is no longer ready")

    def check_ready(self) -> None:
        self._submit(self._ping())

    async def _exchange(self, event: InputEnvelope) -> ActionBatch:
        response = await self._round_trip({"event": json.loads(event.to_json())})
        if set(response) == {"request_id", "error"}:
            raise RuntimeError(str(response["error"])[:4_000])
        if set(response) != {"request_id", "batch"}:
            raise ValueError("strategy response has unexpected fields")
        batch = ActionBatch.from_json(json.dumps(response["batch"]))
        if len(batch) > self.max_actions:
            raise ValueError("strategy exceeded max_actions_per_callback")
        return batch

    def _handle(self, event: InputEnvelope) -> ActionBatch:
        return self._submit(self._exchange(event))

    async def _shutdown(self) -> None:
        try:
            process = getattr(self, "_process", None)
            if process is not None:
                try:
                    async with asyncio.timeout(5):
                        await process.terminate()
                        await process.wait()
                except TimeoutError:
                    await process.kill()
        finally:
            try:
                task = getattr(self, "_stderr_task", None)
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                await self._stack.aclose()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._submit(self._shutdown())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join()

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
