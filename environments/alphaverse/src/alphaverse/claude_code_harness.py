"""Claude Code ACP harness with the isolated Alphaverse player workspace."""

from __future__ import annotations

from verifiers.v1.acp import ACPConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import HarnessSession
from verifiers.v1.harnesses.claude_code import ClaudeCodeHarness
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

from alphaverse.acp_harness import ArtifactExportSession
from alphaverse.eval_harness import (
    install_capture_session,
    install_player_workspace,
    install_role_workspace,
)


class AlphaverseClaudeCodeHarness(ClaudeCodeHarness):
    """Claude Code plus the public player kit and episode capture credential."""

    NETWORK_TOOLS = ("WebFetch", "WebSearch")

    async def setup(self, runtime: Runtime) -> None:
        await super().setup(runtime)
        await install_player_workspace(runtime)

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        await install_role_workspace(data, runtime)
        await install_capture_session(trace, runtime, mcp_urls)
        config = await super().prepare_acp(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
        )
        meta = config.session_meta or {}
        claude = meta.setdefault("claudeCode", {})
        options = claude.setdefault("options", {})
        disabled = options.setdefault("disallowedTools", [])
        options["disallowedTools"] = list(dict.fromkeys([*disabled, *self.NETWORK_TOOLS]))
        config.session_meta = meta
        return config

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        inner = await super().session(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
        )
        return ArtifactExportSession(inner)  # type: ignore[return-value]


__all__ = ["AlphaverseClaudeCodeHarness"]
