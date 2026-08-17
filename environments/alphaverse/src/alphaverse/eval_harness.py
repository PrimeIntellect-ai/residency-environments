"""Leak-free Alphaverse workspace layered on Verifiers' bash harness."""

from __future__ import annotations

import json
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.bash import BashHarness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

from alphaverse.artifact_egress import export_terminal_artifacts
from alphaverse.prop_trader import (
    PROP_PARTICIPANT_ID,
    prop_baseline_source,
    prop_role_readme,
)

_WORKSPACE = Path(__file__).resolve().parent / "agent_workspace"
_WORKSPACE_FILES = ("README.md", "API.md", "market_capture.py")
_SESSION_FILE = ".alphaverse-session.json"


async def install_player_workspace(runtime: Runtime) -> None:
    """Install the public player kit without exposing implementation source."""

    for name in _WORKSPACE_FILES:
        await runtime.write(name, (_WORKSPACE / name).read_bytes())


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


async def install_role_workspace(data: TaskData, runtime: Runtime) -> None:
    """Install role-private seed files after task data identifies the account."""

    if getattr(data, "participant_id", None) != PROP_PARTICIPANT_ID:
        return
    seed_profile = str(getattr(data, "prop_seed_profile", "passive"))
    framing = str(getattr(data, "prop_framing", "incumbent"))
    control_scope = str(getattr(data, "prop_control_scope", "full_source"))
    await runtime.write(
        "ROLE.md",
        prop_role_readme(seed_profile, framing, control_scope).encode("utf-8"),
    )
    await runtime.write(
        "strategy.py",
        prop_baseline_source(seed_profile).encode("utf-8"),
    )


class AlphaverseHarness(BashHarness):
    """Verifiers bash/edit/MCP harness with only the public player kit."""

    async def setup(self, runtime: Runtime) -> None:
        await super().setup(runtime)
        await install_player_workspace(runtime)

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
        await install_role_workspace(data, runtime)
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
    "install_player_workspace",
    "install_role_workspace",
]
