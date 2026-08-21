"""Shared lifecycle helpers for native ACP coding harnesses."""

from __future__ import annotations

from verifiers.v1.harness import HarnessSession
from verifiers.v1.types import Messages

from alphaverse.artifact_egress import export_terminal_artifacts


class ArtifactExportSession:
    """Delegate a native session and export terminal artifacts after each turn."""

    def __init__(self, inner: HarnessSession) -> None:
        self.inner = inner

    async def turn(self, messages: Messages | None = None) -> None:
        await self.inner.turn(messages)
        await export_terminal_artifacts(self.inner.trace, self.inner.mcp_urls)

    async def close(self) -> None:
        await self.inner.close()


__all__ = ["ArtifactExportSession"]
