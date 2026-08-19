from __future__ import annotations

from dataclasses import dataclass
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
    from ..sensors.depth import DepthSensor


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
    depth_sensor: Optional["DepthSensor"] = None
    tick_hook: Optional[Callable[[], object]] = None

    def tick(self, n: int) -> int:
        frame = 0
        for _ in range(max(0, int(n))):
            if self.tick_hook is not None:
                result = self.tick_hook()
                if isinstance(result, int):
                    frame = result
            else:
                frame = self.world.tick()
        return frame
