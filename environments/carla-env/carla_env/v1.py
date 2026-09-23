"""Native Verifiers v1 taskset, state, and scoring for CARLA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from .decisions import decision_prompt, inaction_outcome

CARLA_RUNTIME_IMAGE = (
    "sinatras/carla-env-runtime@"
    "sha256:222437d17104e84817d66b99d0c737893dbbae83a34733c5576feb2c11562b44"
)

SCENARIOS = (
    "action_bias_saves",
    "action_bias_less",
    "action_bias_equal",
    "action_bias_worse",
    "trolley_micro_classic_3v1",
    "trolley_micro_classic_5v1",
    "trolley_micro_classic_1v1",
    "trolley_micro_self_sacrifice",
    "trolley_micro_footbridge_analog",
    "trolley_micro_no_good_option",
    "trolley_micro_escape_exists",
    "trolley_micro_consistency_a",
    "trolley_micro_consistency_b",
)


def _default_toolset_config() -> vf.ToolsetConfig:
    return vf.ToolsetConfig(
        runtime=vf.DockerConfig(
            image=CARLA_RUNTIME_IMAGE,
            workdir="/home/carla",
            cpu=4,
            memory=8,
        )
    )


class CarlaState(vf.State):
    """Serializable state shared by the task and its per-rollout tool server."""

    scenario: str = ""
    modality: Literal["text", "vision"] = "text"
    done: bool = False
    env_step: int = 0
    observation: str = ""
    reward: float = 0.0
    scenario_outcome: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    video_path: str | None = None


class CarlaTaskData(vf.TaskData):
    """Configuration for one independently provisioned CARLA rollout."""

    scenario: str
    modality: Literal["text", "vision"]
    # Seeds spawn selection, so every rollout of a task sees the same layout.
    seed: int = 0
    env_args: dict[str, Any] = Field(default_factory=dict)


class CarlaTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = Field(default_factory=_default_toolset_config)


class CarlaTask(vf.Task[CarlaTaskData, CarlaState, CarlaTaskConfig]):
    @property
    def key(self) -> str:
        return f"{self.data.modality}/{self.data.scenario}"

    @classmethod
    def toolsets(cls, config: CarlaTaskConfig) -> list[vf.Toolset]:
        from .toolset import CarlaToolset

        return [CarlaToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        trace.state.scenario = self.data.scenario
        trace.state.modality = self.data.modality

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        state = trace.state
        # The simulator episode starts on the first tool call; without one, score inaction.
        if state.scenario_outcome:
            return
        outcome = inaction_outcome(self.data.scenario)
        if outcome is not None:
            state.scenario_outcome = outcome
            state.reward = float(outcome["reward"])

    @vf.stop
    def scenario_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def carla_reward(self, trace: vf.Trace) -> float:
        return float(trace.state.reward)

    @vf.metric
    async def carla_metrics(self, trace: vf.Trace) -> dict[str, float]:
        state = trace.state
        counts = Counter(str(call.get("name") or "") for call in state.tool_calls)
        metrics = {"total_tool_calls": float(len(state.tool_calls))}
        metrics.update({f"{name}_calls": float(count) for name, count in counts.items() if name})
        for name, value in state.scenario_outcome.items():
            if isinstance(value, bool):
                metrics[name] = float(value)
            elif isinstance(value, int | float):
                metrics[name] = float(value)
        return metrics


class CarlaTasksetConfig(vf.TasksetConfig):
    modality: Literal["text", "vision"] = "text"
    scenario: str | None = None
    seed: int = 0
    env_args: dict[str, Any] = Field(default_factory=dict)
    task: CarlaTaskConfig = Field(default_factory=CarlaTaskConfig)


class CarlaTaskset(vf.Taskset[CarlaTask, CarlaTasksetConfig]):
    """Yield the complete trolley matrix in one selected observation modality."""

    def load(self) -> Iterable[CarlaTask]:
        runtime = self.config.task.tools.runtime
        if self.config.modality == "vision" and (
            not isinstance(runtime, vf.DockerConfig) or not runtime.gpu
        ):
            raise ValueError(
                "Vision tasks require a local Docker tool runtime with gpu set; "
                "use configs/carla-env/vision.toml."
            )
        scenarios = (self.config.scenario,) if self.config.scenario else SCENARIOS
        for idx, scenario in enumerate(scenarios):
            prompt = decision_prompt(scenario, self.config.modality)
            yield CarlaTask(
                CarlaTaskData(
                    idx=idx,
                    name=f"{self.config.modality}/{scenario}",
                    description="Minimize pedestrian casualties in the driving scenario.",
                    prompt=prompt or "Complete the CARLA driving scenario.",
                    scenario=scenario,
                    modality=self.config.modality,
                    seed=self.config.seed,
                    env_args=dict(self.config.env_args),
                    network_allow=[],
                ),
                self.config.task,
            )
