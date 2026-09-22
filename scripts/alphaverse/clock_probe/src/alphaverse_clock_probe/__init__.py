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
    result = request("tools/call", {"name":name, "arguments":arguments})
    if result.get("isError"):
        raise RuntimeError(result["content"])
    return json.loads(result["content"][0]["text"])
call("product_terms", {})
initial = call("wait", {"duration_ns":0, "wake_on_alert":False})["market_time"]
after_delay = call("wait", {"duration_ns":0, "wake_on_alert":False})["market_time"]
assert after_delay == initial, (initial, after_delay)
call("deploy_strategy", {"source":"from alphaverse.strategy import Strategy\nclass StrategyImpl(Strategy): pass\n",
    "entrypoint":"strategy:StrategyImpl"})
call("submit_limit_order", {"client_order_id":"clock-probe-" + side, "side":side,
    "price":11000 if side == "buy" else 1, "quantity":quantity})
after_wait = call("wait", {"duration_ns":2_000_000_000, "wake_on_alert":False})["market_time"]
assert after_wait == initial + 2_000_000_000, (initial, after_wait)
capture = call("capture_market_data", {"feed":"levels", "after_cursor":0, "limit":100})
account = call("account", {})
if exit_mode == "explicit":
    terminal = call("terminate_session", {})
    assert terminal["time_mode"] == "manual", terminal["time_mode"]
elif exit_mode == "horizon":
    call("wait", {"duration_ns":20_000_000_000, "wake_on_alert":False})
observations = {"capture":capture, "account":account, "initial":initial, "after_wait":after_wait}
print(json.dumps({"initial_market_time":initial, "after_delay_market_time":after_delay,
    "after_wait_market_time":after_wait, "time_mode":"manual", "exit_mode":exit_mode,
    "side":side, "account_before_closeout":account,
    "observations_sha256":hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()}))
"""


class ClockProbeHarnessConfig(HarnessConfig):
    delay_seconds: float = Field(default=0.0, ge=0, le=2)
    exit_mode: Literal["explicit", "harness", "horizon"] = "explicit"
    side: Literal["buy", "sell"] = "buy"
    quantity: int = Field(default=1, ge=1)


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
                ],
                {},
            )
            if result.exit_code != 0:
                raise RuntimeError(f"clock probe failed: {result.stderr[-2000:]}")
            evidence = json.loads(result.stdout)
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
