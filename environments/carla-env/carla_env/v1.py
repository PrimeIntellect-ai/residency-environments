"""Native Verifiers v1 taskset, state, and scoring for CARLA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.harnesses.null import NullHarness

from .decisions import decision_prompt, inaction_outcome
from .procedural import procedural_inaction_outcome, procedural_prompt

CARLA_RUNTIME_IMAGE = (
    "sinatras/carla-env-runtime@"
    "sha256:602cc9ece967f16b73f4c809aa50ef7c7525a280ebbd1ef3d9f53634a0a23d23"
)

ScenarioFamily = Literal["decision", "maze", "navigation", "free_roam"]

DECISION_SCENARIOS = (
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

SCENARIOS_BY_FAMILY: dict[ScenarioFamily, tuple[str, ...]] = {
    "decision": DECISION_SCENARIOS,
    "maze": ("maze",),
    "navigation": ("navigation_Town10HD_v10_p20",),
    "free_roam": ("free_roam_Town10HD_v10_p20",),
}

SCENARIOS = tuple(
    scenario for family_scenarios in SCENARIOS_BY_FAMILY.values() for scenario in family_scenarios
)


CARLA_WORKDIR = "/home/carla"
CARLA_RESOURCES = vf.TaskResources(cpu=4, memory=8, disk=40)


def _default_toolset_config() -> vf.ToolsetConfig:
    # Prime sandboxes do not expose ports, so by default the tool server runs inside the
    # agent's sandbox, which each task provisions from the runtime image.
    return vf.ToolsetConfig(colocated=True)


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
    family: ScenarioFamily
    modality: Literal["text", "vision"]
    # Seeds spawn selection and scenario randomness, so every rollout of a task sees
    # the same layout.
    seed: int = 0
    env_args: dict[str, Any] = Field(default_factory=dict)


class CarlaTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = Field(default_factory=_default_toolset_config)


class CarlaTask(vf.Task[CarlaTaskData, CarlaState, CarlaTaskConfig]):
    @property
    def key(self) -> str:
        return _task_name(self.data.modality, self.data.family, self.data.scenario, self.data.seed)

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
        if self.data.family == "decision":
            outcome = inaction_outcome(self.data.scenario)
        else:
            outcome = procedural_inaction_outcome(self.data.scenario)
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


class CarlaHarness(NullHarness):
    """Default harness: the model acts only through the task's CARLA tools.

    The colocated tool server shares the agent's sandbox, so a harness that executes
    code there could reach the CARLA RPC port.
    """


class CarlaTasksetConfig(vf.TasksetConfig):
    modality: Literal["text", "vision"] = "text"
    scenario: str | None = None
    families: list[ScenarioFamily] = Field(
        default_factory=lambda: ["decision", "maze", "navigation", "free_roam"],
        min_length=1,
    )
    # Procedural families yield one task per seed; decision tasks use the first seed.
    seeds: list[int] = Field(default_factory=lambda: [0, 1], min_length=1)
    env_args: dict[str, Any] = Field(default_factory=dict)
    task: CarlaTaskConfig = Field(default_factory=CarlaTaskConfig)


def _scenario_family(scenario: str) -> ScenarioFamily:
    if scenario.startswith(("action_bias", "bias_", "trolley_micro_")):
        return "decision"
    if scenario.startswith("maze"):
        return "maze"
    if scenario.startswith("navigation"):
        return "navigation"
    if scenario.startswith("free_roam"):
        return "free_roam"
    raise ValueError(f"Unknown CARLA scenario family: {scenario}")


def _task_name(modality: str, family: ScenarioFamily, scenario: str, seed: int) -> str:
    """Decision tasks are fixed, so only procedural task names carry their seed."""
    suffix = "" if family == "decision" else f"/seed-{seed}"
    return f"{modality}/{scenario}{suffix}"


_DESCRIPTIONS: dict[ScenarioFamily, str] = {
    "decision": "Minimize pedestrian casualties in the driving scenario.",
    "maze": "Reach a procedurally selected hidden destination.",
    "navigation": "Navigate safely to a procedurally selected destination.",
    "free_roam": "Explore the procedurally generated traffic scene safely.",
}


def _task_prompt(family: ScenarioFamily, scenario: str, modality: str) -> str:
    if family != "decision":
        return procedural_prompt(family, modality)
    prompt = decision_prompt(scenario, modality)
    if prompt is None:
        raise ValueError(f"Unknown CARLA decision scenario: {scenario}")
    return prompt


class CarlaTaskset(vf.Taskset[CarlaTask, CarlaTasksetConfig]):
    """Yield fixed decision tasks and reproducibly generated driving tasks."""

    def _task_specs(self) -> Iterable[tuple[ScenarioFamily, str, int]]:
        seeds = tuple(dict.fromkeys(self.config.seeds))
        if self.config.scenario:
            specs = [(_scenario_family(self.config.scenario), self.config.scenario)]
        else:
            specs = [
                (family, scenario)
                for family in dict.fromkeys(self.config.families)
                for scenario in SCENARIOS_BY_FAMILY[family]
            ]
        for family, scenario in specs:
            for seed in seeds[:1] if family == "decision" else seeds:
                yield family, scenario, seed

    def load(self) -> Iterable[CarlaTask]:
        runtime = self.config.task.tools.runtime
        if self.config.modality == "vision" and (
            not isinstance(runtime, vf.DockerConfig) or not runtime.gpu
        ):
            raise ValueError(
                "Vision tasks require a local Docker tool runtime with gpu set; "
                "use configs/carla-env/vision.toml."
            )
        # A colocated tool server needs CARLA in the agent's sandbox; a separate tool
        # runtime brings its own image instead.
        colocated = self.config.task.tools.colocated
        for idx, (family, scenario, seed) in enumerate(self._task_specs()):
            yield CarlaTask(
                CarlaTaskData(
                    idx=idx,
                    name=_task_name(self.config.modality, family, scenario, seed),
                    description=_DESCRIPTIONS[family],
                    prompt=_task_prompt(family, scenario, self.config.modality),
                    scenario=scenario,
                    family=family,
                    seed=seed,
                    modality=self.config.modality,
                    env_args=dict(self.config.env_args),
                    image=CARLA_RUNTIME_IMAGE if colocated else None,
                    workdir=CARLA_WORKDIR if colocated else None,
                    resources=CARLA_RESOURCES if colocated else vf.TaskResources(),
                    network_allow=[],
                ),
                self.config.task,
            )
