"""Single-player orchestration with market closeout before scoring."""

from __future__ import annotations

import verifiers.v1 as vf

from alphaverse.artifact_egress import finalize_episode, release_trading_runtimes
from alphaverse.verifiers_v1 import AlphaverseTask


class AlphaverseEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()


async def finalize_interaction(interaction) -> None:
    # The pinned Verifiers Interaction exposes its live tool URLs on its rollout.
    urls = interaction._run._urls
    try:
        await finalize_episode(interaction.trace, urls)
    finally:
        await release_trading_runtimes(interaction.trace, urls)


class AlphaverseEnv(vf.Env[AlphaverseEnvConfig]):
    async def run(self, task: AlphaverseTask, agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn()
            await finalize_interaction(interaction)
