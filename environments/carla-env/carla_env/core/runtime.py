from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import carla

from ..logging import get_logger
from ..sensors import CollisionSensor, TextSensor
from .actors import ActorManager
from .client import CarlaClient
from .world import WorldManager

logger = get_logger("core.runtime")

if TYPE_CHECKING:
    from ..sensors.camera import CameraSensor


@dataclass
class CarlaRuntime:
    """
    Bundle of CARLA objects for a single rollout.

    Stored in `state["carla"]` and treated as opaque by tools (they use state injection).
    """

    client: CarlaClient
    world: WorldManager
    actors: ActorManager
    ego_vehicle: carla.Vehicle
    text_sensor: TextSensor
    collision_sensor: CollisionSensor
    camera_sensor: Optional["CameraSensor"] = None
    tick_hook: Optional[Callable[[], object]] = None
    # Called after every tick, so scenarios can sample the ego path inside long tool calls.
    tick_listeners: list[Callable[[], None]] = field(default_factory=list)
    # Episode clock, started by start_episode_clock; None limits are off.
    episode_ticks: int = 0
    max_episode_ticks: int | None = None
    wall_deadline: float | None = None

    @property
    def delta_seconds(self) -> float:
        return float(getattr(self.world.config, "fixed_delta_seconds", 0.05) or 0.05)

    @property
    def sim_seconds(self) -> float:
        """Simulated time since the episode started."""
        return self.episode_ticks * self.delta_seconds

    def start_episode_clock(
        self, max_sim_seconds: float | None, max_wall_seconds: float | None
    ) -> None:
        self.episode_ticks = 0
        self.max_episode_ticks = (
            None
            if max_sim_seconds is None
            else max(1, math.ceil(max_sim_seconds / self.delta_seconds - 1e-9))
        )
        self.wall_deadline = (
            None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
        )

    def time_limit_reached(self) -> str | None:
        """Name of the episode time limit that has been reached, if any."""
        if self.max_episode_ticks is not None and self.episode_ticks >= self.max_episode_ticks:
            return "max_sim_seconds"
        if self.wall_deadline is not None and time.monotonic() >= self.wall_deadline:
            return "max_wall_seconds"
        return None

    def tick(self, n: int) -> int:
        frame = 0
        for _ in range(max(0, int(n))):
            # Past an episode time limit the world stops, even inside a tool call.
            if self.time_limit_reached() is not None:
                break
            if self.tick_hook is not None:
                result = self.tick_hook()
                frame = result if isinstance(result, int) else 0
            else:
                frame = self.world.tick()
            # A failed tick returns frame 0 and does not advance the episode clock.
            if frame:
                self.episode_ticks += 1
            for listener in self.tick_listeners:
                listener()
        return frame
