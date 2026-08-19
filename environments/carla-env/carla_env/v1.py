"""Verifiers v1 adapter for the CARLA environment."""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Callable, Iterable
from typing import Any

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.agent import Agents
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.task import Task, TaskData
from verifiers.v1.trace import Error, Trace

from .bridge import legacy_client, legacy_output_to_trace


class CarlaTaskData(TaskData):
    """One indexed rollout from the configured legacy dataset."""


class CarlaTask(Task[CarlaTaskData]):
    """Task marker used by the v1 taskset loader."""


class CarlaTasksetConfig(TasksetConfig):
    env_args: dict[str, Any] = Field(default_factory=dict)
    """Arguments forwarded unchanged to ``load_environment``."""

    num_tasks: int = Field(1, ge=1)
    """Number of rows exposed from the configured dataset."""


class CarlaTaskset(vf.Taskset[CarlaTask, CarlaTasksetConfig]):
    """Expose CARLA episodes through the Verifiers v1 taskset interface."""

    def load(self) -> Iterable[CarlaTask]:
        for idx in range(self.config.num_tasks):
            yield CarlaTask(
                CarlaTaskData(
                    idx=idx,
                    name=f"carla-{idx}",
                    description="Complete the configured CARLA driving scenario.",
                )
            )


class CarlaV1EnvConfig(EnvConfig):
    agent: AgentConfig = AgentConfig()


class CarlaV1Env(vf.Env[CarlaV1EnvConfig]):
    """Run the established stateful CARLA rollout through the official v1 bridge."""

    def __init__(self, config: CarlaV1EnvConfig) -> None:
        super().__init__(config)
        self._legacy_env = None
        self._dataset = None
        self._clients: dict[tuple[str, str], Any] = {}

    async def start(self) -> None:
        from .env import load_environment

        taskset_config = self.taskset.config
        if not isinstance(taskset_config, CarlaTasksetConfig):
            raise TypeError("CarlaV1Env requires CarlaTasksetConfig")
        self._legacy_env = load_environment(**taskset_config.env_args)
        try:
            self._dataset = self._legacy_env.get_dataset()
        except ValueError:
            self._dataset = self._legacy_env.get_eval_dataset()
        if len(self._dataset) < taskset_config.num_tasks:
            raise ValueError(
                f"num_tasks={taskset_config.num_tasks} exceeds the configured "
                f"dataset length ({len(self._dataset)})"
            )

    async def stop(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result
        self._clients.clear()
        self._dataset = None
        self._legacy_env = None

    async def run(self, task: Task, agents: Agents) -> None:
        raise NotImplementedError("CarlaV1Env owns the bridged rollout lifecycle")

    async def run_episode(
        self,
        task: Task,
        ctx: ModelContext,
        *,
        on_trace: Callable[[Trace], None] | None = None,
        on_discard: Callable[[Trace], None] | None = None,
    ) -> Episode:
        del on_discard
        if self._legacy_env is None or self._dataset is None:
            raise RuntimeError("CarlaV1Env is not serving")
        task_idx = task.data.idx
        if task_idx is None:
            raise ValueError("CARLA tasks require an index")

        key = (ctx.client.model_dump_json(), ctx.model)
        client = self._clients.get(key)
        if client is None:
            client = legacy_client(ctx.client, ctx.model)
            self._clients[key] = client

        try:
            output = await self._legacy_env.run_rollout(
                input=dict(self._dataset[task_idx]),
                client=client,
                model=ctx.model,
                sampling_args=ctx.sampling.model_dump(exclude_none=True),
                state_columns=["trajectory"],
            )
            trace = legacy_output_to_trace(output, task)
            if on_trace is not None:
                on_trace(trace)
            return Episode.of(trace, env=self.config.env_id)
        except Exception as exc:  # noqa: BLE001 - the episode records its boundary failure
            return Episode(
                env=EnvInfo(id=self.config.env_id),
                errors=[
                    Error(
                        type=type(exc).__name__,
                        message=str(exc),
                        traceback=traceback.format_exc(),
                    )
                ],
            )
