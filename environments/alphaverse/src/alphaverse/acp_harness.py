"""Shared lifecycle helpers for native ACP coding harnesses."""

from __future__ import annotations

from verifiers.v1.harness import HarnessSession
from verifiers.v1.types import Messages

from alphaverse.artifact_egress import export_terminal_artifacts, release_trading_runtimes


class ArtifactExportSession:
    """Delegate a native session and export terminal artifacts after each turn."""

    def __init__(self, inner: HarnessSession, mcp_urls: dict[str, str]) -> None:
        self.inner = inner
        self.mcp_urls = dict(mcp_urls)

    async def turn(self, messages: Messages | None = None) -> None:
        await self.inner.turn(messages)
        await export_terminal_artifacts(self.inner.trace, self.mcp_urls)

    async def close(self) -> None:
        try:
            await self.inner.close()
            await export_terminal_artifacts(self.inner.trace, self.mcp_urls)
        finally:
            await release_trading_runtimes(self.inner.trace, self.mcp_urls)


__all__ = ["ArtifactExportSession"]
