"""Bash harness transport needed after task-owned workspace setup."""

from __future__ import annotations

import json

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.bash import BashHarness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

from alphaverse.artifact_egress import export_terminal_artifacts

_SESSION_FILE = ".alphaverse-session.json"


async def install_capture_session(
    trace: Trace,
    runtime: Runtime,
    mcp_urls: dict[str, str] | None = None,
) -> None:
    """Install the private file-capture transport used by market_capture.py."""

    mcp_url = (mcp_urls or {}).get("alphaverse")
    if not isinstance(mcp_url, str) or not mcp_url:
        raise RuntimeError("Alphaverse Toolset URL is unavailable")
    if hasattr(trace.state, "toolset_url"):
        trace.state.toolset_url = mcp_url
    session = {"mcp_url": mcp_url}
    await runtime.write(
        _SESSION_FILE,
        json.dumps(session, sort_keys=True).encode("utf-8"),
    )


class AlphaverseHarness(BashHarness):
    """Bash harness with private capture transport and artifact export."""

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        await install_capture_session(trace, runtime, mcp_urls)
        result = await super().launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )
        await export_terminal_artifacts(trace, mcp_urls)
        return result


__all__ = [
    "AlphaverseHarness",
    "install_capture_session",
]
