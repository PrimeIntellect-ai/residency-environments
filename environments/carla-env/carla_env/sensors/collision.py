from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import carla

from ..compat import safe_stop_sensor
from ..logging import get_logger

logger = get_logger("sensors.collision")


@dataclass
class CollisionEvent:
    frame: int
    other_actor_id: int
    other_actor_type: str


class CollisionSensor:
    """
    CARLA collision sensor wrapper.

    Stores all collision events so reward computation can be done without
    querying CARLA after cleanup.
    """

    def __init__(self, world, actor_manager, parent: carla.Actor):
        self._world = world
        self._actor_manager = actor_manager
        self._parent = parent
        self._sensor: Optional[carla.Sensor] = None
        self._events: list[CollisionEvent] = []
        # Suppress repeated parse-failure warnings after the first occurrence.
        self._other_actor_parse_failures: int = 0
        self._frame_parse_failures: int = 0

    def setup(self) -> None:
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        sensor = self._actor_manager.spawn_sensor(
            "sensor.other.collision",
            transform,
            attach_to=self._parent,
        )
        assert isinstance(sensor, carla.Sensor)
        self._sensor = sensor

        def _on_collision(event: Any) -> None:
            try:
                other = event.other_actor
                other_id = int(getattr(other, "id", -1))
                other_type = str(getattr(other, "type_id", "")) if other else ""
            except Exception as e:  # noqa: BLE001
                self._other_actor_parse_failures += 1
                if self._other_actor_parse_failures == 1:
                    logger.warning(
                        "Failed to extract other_actor from collision event (suppressing further warnings): %s",
                        e,
                    )
                else:
                    logger.debug(
                        "Failed to extract other_actor from collision event", exc_info=True
                    )
                other_id = -1
                other_type = ""
            try:
                frame = int(getattr(event, "frame", -1))
            except Exception as e:  # noqa: BLE001
                self._frame_parse_failures += 1
                if self._frame_parse_failures == 1:
                    logger.warning(
                        "Failed to extract frame from collision event (suppressing further warnings): %s",
                        e,
                    )
                else:
                    logger.debug("Failed to extract frame from collision event", exc_info=True)
                frame = -1
            self._events.append(
                CollisionEvent(frame=frame, other_actor_id=other_id, other_actor_type=other_type)
            )

        sensor.listen(_on_collision)

    def destroy(self) -> None:
        safe_stop_sensor(self._sensor)
        self._sensor = None

    @property
    def events(self) -> list[CollisionEvent]:
        return list(self._events)

    @property
    def collision_count(self) -> int:
        return len(self._events)

    def count_by_prefix(self, type_prefix: str) -> int:
        return sum(1 for e in self._events if e.other_actor_type.startswith(type_prefix))

    def count_unique_by_prefix(self, type_prefix: str) -> int:
        return len(
            {e.other_actor_id for e in self._events if e.other_actor_type.startswith(type_prefix)}
        )
