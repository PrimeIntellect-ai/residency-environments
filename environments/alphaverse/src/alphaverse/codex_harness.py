"""Codex ACP transport restrictions and terminal artifact export."""

from __future__ import annotations

import json

from verifiers.v1.acp import ACPConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import HarnessSession
from verifiers.v1.harnesses.codex import CodexHarness
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

from alphaverse.acp_harness import ArtifactExportSession
from alphaverse.eval_harness import install_capture_session


class AlphaverseCodexHarness(CodexHarness):
    """Codex plus private capture transport and terminal artifact export."""

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
        codex_config = json.loads(config.env["CODEX_CONFIG"])
        codex_config["web_search"] = "disabled"
        config.env["CODEX_CONFIG"] = json.dumps(codex_config)
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


__all__ = ["AlphaverseCodexHarness"]
