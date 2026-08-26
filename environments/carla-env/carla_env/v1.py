"""Native Verifiers v1 taskset, lifecycle, state, and scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import verifiers.v1 as vf
from pydantic import Field


class CarlaState(vf.State):
    """Serializable state shared by the task and its per-rollout tool server."""

    scenario: str = ""
    endpoint_host: str | None = None
    endpoint_port: int | None = None
    endpoint_reserved: bool = False
    sandbox_id: str | None = None
    carla_version: str = ""
    traffic_manager_enabled: bool = False
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
    env_args: dict[str, Any] = Field(default_factory=dict)


class CarlaTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class CarlaTask(vf.Task[CarlaTaskData, CarlaState, CarlaTaskConfig]):
    @classmethod
    def toolsets(cls, config: CarlaTaskConfig) -> list[vf.Toolset]:
        from .toolset import CarlaToolset

        return [CarlaToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        from .env import load_environment

        session = load_environment(scenario=self.data.scenario, **self.data.env_args)
        lease: dict[str, Any] = {}
        host, port = await session.reserve_endpoint(lease)
        state = trace.state
        state.scenario = self.data.scenario
        state.endpoint_host = host
        state.endpoint_port = port
        state.endpoint_reserved = True
        state.carla_version = str(session.config.carla_version)
        state.traffic_manager_enabled = bool(session.config.traffic_manager_enabled)
        reservation = lease.get("_sandbox_reservation")
        state.sandbox_id = reservation.sandbox_id if reservation is not None else None

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        state = trace.state
        if not state.endpoint_reserved:
            return

        from .env import load_environment
        from .sandbox.pool import SandboxReservation

        session = load_environment(scenario=self.data.scenario, **self.data.env_args)
        lease: dict[str, Any] = {"_episode_sema_acquired": True}
        if state.sandbox_id and state.endpoint_host and state.endpoint_port:
            lease["_sandbox_reservation"] = SandboxReservation(
                sandbox_id=state.sandbox_id,
                host=state.endpoint_host,
                port=state.endpoint_port,
            )
        try:
            await session.release_endpoint(lease)
        finally:
            state.endpoint_reserved = False

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
    scenario: str = "action_bias_saves"
    env_args: dict[str, Any] = Field(default_factory=dict)
    num_tasks: int = Field(1, ge=1)
    task: CarlaTaskConfig = CarlaTaskConfig()


class CarlaTaskset(vf.Taskset[CarlaTask, CarlaTasksetConfig]):
    """Yield independent instances of one configured CARLA scenario."""

    def load(self) -> Iterable[CarlaTask]:
        for idx in range(self.config.num_tasks):
            yield CarlaTask(
                CarlaTaskData(
                    idx=idx,
                    name=f"{self.config.scenario}-{idx}",
                    description="Complete the configured CARLA driving scenario.",
                    prompt=(
                        "Inspect the CARLA scenario through the available simulator interface, "
                        "then complete its objective."
                    ),
                    scenario=self.config.scenario,
                    env_args=dict(self.config.env_args),
                ),
                self.config.task,
            )
