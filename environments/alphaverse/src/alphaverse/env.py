"""Single-player orchestration with market closeout before scoring."""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.harnesses.bash.harness import BashHarnessConfig

from alphaverse.artifact_egress import finalize_episode, release_trading_runtimes
from alphaverse.verifiers_v1 import AlphaverseTask, AlphaverseTasksetConfig


class AlphaverseEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig().model_copy(update={"harness": BashHarnessConfig(id="alphaverse")})
    episode_time_limit_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_tool_deadlines(self):
        harness = self.agent.harness
        if (
            isinstance(self.taskset, AlphaverseTasksetConfig)
            and harness is not None
            and harness.tool_timeout
            <= self.taskset.task.toolset.tool_timeout_seconds + self.taskset.task.toolset.cleanup_timeout_seconds + 5
        ):
            raise ValueError(
                "agent.harness.tool_timeout must exceed the server tool + cleanup + 5-second unwind budget"
            )
        return self


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
            limit = asyncio.timeout(self.config.episode_time_limit_seconds)
            try:
                async with limit:
                    await interaction.turn()
            except TimeoutError:
                if not limit.expired():
                    raise
                interaction.trace.state.limit_reached = "episode_time"
                interaction.trace.stop("episode_time_limit")
            await finalize_interaction(interaction)
