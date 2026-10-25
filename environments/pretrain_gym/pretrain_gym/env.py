"""The single-agent env, with the agent's turn limit copied into the task so its prompt can state it."""

import verifiers.v1 as vf
from pydantic import model_validator

from .config import PretrainTasksetConfig


class CurationEnvConfig(vf.SingleAgentEnvConfig):
    taskset: PretrainTasksetConfig = PretrainTasksetConfig()

    @model_validator(mode="after")
    def _copy_turn_limit(self) -> "CurationEnvConfig":
        task = self.taskset.task.model_copy(update={"max_turns": self.agent.max_turns})
        return self.model_copy(update={"taskset": self.taskset.model_copy(update={"task": task})})


class CurationEnv(vf.Env[CurationEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)
