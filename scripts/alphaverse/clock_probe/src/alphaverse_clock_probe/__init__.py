"""Exercise real native runtimes and compare latency-independent market output."""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import Literal

from alphaverse.artifact_egress import export_terminal_artifacts, release_trading_runtimes
from pydantic import Field
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness

PROGRAM = r"""import hashlib
import json
import sys
import time
import urllib.request

url, delay, exit_mode, side = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
quantity = int(sys.argv[5])
strategy_mode, require_yield = sys.argv[6], sys.argv[7] == "True"
pause_seconds = float(sys.argv[8])
expect_tool_error = sys.argv[9] == "True"
wait_duration_ns = int(sys.argv[10])
timings = []
def request(method, params):
    data = json.dumps({"jsonrpc":"2.0", "id":1, "method":method, "params":params}).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type":"application/json", "Accept":"application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["result"]
request("initialize", {"protocolVersion":"2025-03-26", "capabilities":{},
    "clientInfo":{"name":"clock-probe", "version":"1"}})
names = {t["name"] for t in request("tools/list", {})["tools"]}
def call(name, arguments):
    time.sleep(delay)
    name = name if name in names else "alphaverse_" + name
    started = time.monotonic()
    result = request("tools/call", {"name":name, "arguments":arguments})
    timings.append({"tool":name, "seconds":time.monotonic() - started})
    if result.get("isError"):
        raise RuntimeError(result["content"])
    return json.loads(result["content"][0]["text"])
call("product_terms", {})
initial = call("wait", {"duration_ns":0, "wake_on_alert":False})["market_time"]
after_delay = call("wait", {"duration_ns":0, "wake_on_alert":False})["market_time"]
assert after_delay == initial, (initial, after_delay)
source = "from alphaverse.strategy import Strategy\nclass StrategyImpl(Strategy): pass\n"
if strategy_mode == "busy":
    source = "from alphaverse.strategy import Strategy\nclass StrategyImpl(Strategy):\n    def on_start(self, ctx, event):\n        while True: pass\n"
try:
    call("deploy_strategy", {"source":source, "entrypoint":"strategy:StrategyImpl"})
except RuntimeError as exc:
    if not expect_tool_error:
        raise
    print(json.dumps({"expected_tool_error":str(exc), "tool_timings":timings}))
    sys.exit(0)
if expect_tool_error:
    raise RuntimeError("expected a deployment deadline error")
call("submit_limit_order", {"client_order_id":"clock-probe-" + side, "side":side,
    "price":11000 if side == "buy" else 1, "quantity":quantity})
yield_count = 0
def wait_to(target):
    global yield_count
    for _ in range(1000):
        response = call("wait", {"until_ns":target, "wake_on_alert":False})
        if not response.get("yielded_for_budget"):
            return response["market_time"]
        yield_count += 1
        assert response["requested_until"] == target, response
    raise RuntimeError("wait did not finish within 1000 slices")
after_wait = wait_to(initial + wait_duration_ns)
assert after_wait == initial + wait_duration_ns, (initial, after_wait)
if require_yield:
    assert yield_count > 0, "expected a bounded wait slice"
strategy_status = call("strategy_status", {})
if strategy_mode == "busy":
    assert not strategy_status["active"] and "deadline" in strategy_status["fault"], strategy_status
capture = call("capture_market_data", {"feed":"levels", "after_cursor":0, "limit":100})
account = call("account", {})
time.sleep(pause_seconds)
if exit_mode == "explicit":
    terminal = call("terminate_session", {})
    assert terminal["time_mode"] == "manual", terminal["time_mode"]
elif exit_mode == "horizon":
    wait_to(20_000_000_000)
observations = {"capture":capture, "account":account, "initial":initial, "after_wait":after_wait}
print(json.dumps({"initial_market_time":initial, "after_delay_market_time":after_delay,
    "after_wait_market_time":after_wait, "time_mode":"manual", "exit_mode":exit_mode,
    "side":side, "account_before_closeout":account,
    "yield_count":yield_count, "strategy_status":strategy_status, "tool_timings":timings,
    "observations_sha256":hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()}))
"""


class ClockProbeHarnessConfig(HarnessConfig):
    delay_seconds: float = Field(default=0.0, ge=0, le=2)
    exit_mode: Literal["explicit", "harness", "horizon"] = "explicit"
    side: Literal["buy", "sell"] = "buy"
    quantity: int = Field(default=1, ge=1)
    strategy_mode: Literal["noop", "busy"] = "noop"
    require_yield: bool = False
    pause_seconds: float = Field(default=0.0, ge=0, le=300)
    expect_tool_error: bool = False
    wait_duration_ns: int = Field(default=2_000_000_000, gt=0)


class ClockProbeHarness(Harness[ClockProbeHarnessConfig]):
    SUPPORTS_MCP = True
    EXECUTES_CODE = True
    NEEDS_CONTAINER = True
    SUPPORTS_RESUME = True

    async def setup(self, runtime) -> None:
        await runtime.write("/tmp/clock-probe.py", PROGRAM.encode())

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        try:
            result = await runtime.run(
                [
                    "python",
                    "/tmp/clock-probe.py",
                    mcp_urls["alphaverse"],
                    str(self.config.delay_seconds),
                    self.config.exit_mode,
                    self.config.side,
                    str(self.config.quantity),
                    self.config.strategy_mode,
                    str(self.config.require_yield),
                    str(self.config.pause_seconds),
                    str(self.config.expect_tool_error),
                    str(self.config.wait_duration_ns),
                ],
                {},
            )
            if result.exit_code != 0:
                raise RuntimeError(f"clock probe failed: {result.stderr[-2000:]}")
            evidence = json.loads(result.stdout)
            if self.config.expect_tool_error:
                trace.info["clock_probe"] = evidence
                return result
            if self.config.exit_mode == "harness":
                trace.info["clock_probe"] = evidence
                return result
            directory = await export_terminal_artifacts(trace, mcp_urls)
            if directory is None:
                raise RuntimeError("clock probe requires streamed terminal artifacts")
            with gzip.open(directory / "canonical-events.ndjson.gz", "rb") as stream:
                canonical = stream.read()
            evidence["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
            evidence["canonical_events"] = len(canonical.splitlines())
            evidence["delay_seconds"] = self.config.delay_seconds
            trace.info["clock_probe"] = evidence
            return result
        finally:
            if self.config.exit_mode != "harness":
                await release_trading_runtimes(trace, mcp_urls)


__all__ = ["ClockProbeHarness"]
