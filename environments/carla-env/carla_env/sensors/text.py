from __future__ import annotations

import math
from dataclasses import dataclass

import carla


@dataclass
class TextObs:
    text: str
    ego_speed_mps: float
    ego_speed_kmh: float
    ego_lane_id: int | None
    ego_road_id: int | None


class TextSensor:
    """
    Synthetic "sensor" that produces a compact text observation.

    We intentionally keep this compact and scenario-agnostic; scenario tools can
    expose extra info via dedicated tools (e.g., get_goal_info for Maze).
    """

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Actor,
        radius_m: float = 50.0,
        max_actors: int = 8,
    ):
        self._world = world
        self._ego = ego_vehicle
        self._radius_m = float(radius_m)
        self._max_actors = int(max_actors)
        self._map = world.get_map()

    def observe(self) -> TextObs:
        ego = self._ego
        if ego is None or not ego.is_alive:
            return TextObs(
                text="Ego vehicle not available.",
                ego_speed_mps=0.0,
                ego_speed_kmh=0.0,
                ego_lane_id=None,
                ego_road_id=None,
            )

        loc = ego.get_location()
        vel = ego.get_velocity()
        speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = speed_mps * 3.6

        wp = self._map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            wp = self._map.get_waypoint(loc, project_to_road=True)

        lane_id = int(getattr(wp, "lane_id", 0)) if wp is not None else None
        road_id = int(getattr(wp, "road_id", 0)) if wp is not None else None

        lines: list[str] = []
        lines.append(f"Ego speed: {speed_kmh:.1f} km/h")
        if wp is not None:
            lines.append(f"Lane: lane_id={lane_id} road_id={road_id}")
        else:
            lines.append("Lane: unknown (off-road)")

        # Nearby actors summary
        actors = self._world.get_actors()
        nearby: list[tuple[float, str]] = []
        for actor in actors:
            try:
                if actor.id == ego.id:
                    continue
                tid = str(getattr(actor, "type_id", ""))
                if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                    continue
                dist = float(actor.get_location().distance(loc))
                if dist <= self._radius_m:
                    nearby.append((dist, tid))
            except Exception:
                continue
        nearby.sort(key=lambda x: x[0])

        if nearby:
            lines.append(f"Nearby actors ({min(len(nearby), self._max_actors)}):")
            for dist, tid in nearby[: self._max_actors]:
                kind = "ped" if tid.startswith("walker.") else "veh"
                lines.append(f"  - {kind} {dist:.1f}m {tid}")
        else:
            lines.append("Nearby actors: none")

        return TextObs(
            text="\n".join(lines),
            ego_speed_mps=speed_mps,
            ego_speed_kmh=speed_kmh,
            ego_lane_id=lane_id,
            ego_road_id=road_id,
        )
