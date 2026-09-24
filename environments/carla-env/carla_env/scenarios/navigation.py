from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import carla

from ..logging import get_logger
from ..procedural import NAVIGATION_SUCCESS_RADIUS_M
from .base import BaseScenario, ScenarioConfig

logger = get_logger("scenarios.navigation")

WEATHER_PRESETS: List[str] = [
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "HardRainNoon",
    "SoftRainNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "HardRainSunset",
    "SoftRainSunset",
]


def _coerce_goal_location(goal: Tuple[float, float, float]) -> carla.Location:
    return carla.Location(
        x=float(goal[0]),
        y=float(goal[1]),
        z=float(goal[2]),
    )


def _route_length_m(
    origin: carla.Location,
    destination: carla.Location,
    route: list[tuple[object, object]] | list[object],
) -> float:
    points: list[carla.Location] = [origin]
    for segment in route:
        waypoint = segment[0] if isinstance(segment, tuple) else segment
        transform = getattr(waypoint, "transform", None)
        location = getattr(transform, "location", None)
        if location is not None:
            points.append(location)
    points.append(destination)

    total = 0.0
    prev = points[0]
    for location in points[1:]:
        total += float(prev.distance(location))
        prev = location
    return total


@dataclass
class NavigationConfig(ScenarioConfig):
    map_name: Optional[str] = None
    num_npc_vehicles: int = 0
    num_pedestrians: int = 0
    success_radius: float = NAVIGATION_SUCCESS_RADIUS_M
    random_goal: bool = True
    goal_location: Optional[Tuple[float, float, float]] = None
    route_distance_min: float = 100.0
    route_distance_max: float = 500.0
    max_steps: int = 500
    auto_observe: bool = True
    idle_ticks: int = 1
    enable_vision: bool = False
    vision_only: bool = False


class NavigationScenario(BaseScenario[NavigationConfig]):
    """Configurable autonomous driving scenario."""

    def __init__(self, config: NavigationConfig):
        super().__init__(config)
        self._configured_weather = config.weather
        self._rng = random.Random()

    def supports_goal_info(self) -> bool:
        return True

    def spawn_requirements(self) -> Dict[str, Any]:
        reqs: Dict[str, Any] = {
            "require_left": False,
            "require_right": False,
            "min_forward_m": 10.0,
        }
        if self.config.map_name:
            reqs["map_name"] = self.config.map_name
        return reqs

    def reset(self, state: Any) -> None:
        self._rng.seed(self.config.seed)
        state.setdefault("scenario_state", {})
        state["scenario_state"]["navigation"] = {
            "initial_route_distance": None,
            "best_distance_m": None,
            "collision_count": 0,
        }
        if self._configured_weather == "random":
            self.config.weather = self._rng.choice(WEATHER_PRESETS)

    def setup(self, state: Any) -> None:
        runtime = state["carla"]
        navigation_state = state["scenario_state"]["navigation"]
        carla_map = runtime.world.map
        spawn_points = list(carla_map.get_spawn_points())
        if not spawn_points:
            raise RuntimeError("NavigationScenario: map has no spawn points")
        ego_location = runtime.ego_vehicle.get_transform().location

        goal_loc = self._pick_goal(ego_location, spawn_points, carla_map)
        state.setdefault("scenario_data", {})
        state["scenario_data"]["goal_location"] = goal_loc
        goal_spawn_location = _coerce_goal_location(goal_loc)

        initial_distance = float(goal_spawn_location.distance(ego_location))
        navigation_state["initial_route_distance"] = initial_distance
        navigation_state["best_distance_m"] = initial_distance
        navigation_state["goal_reached"] = False
        runtime.tick_listeners.append(lambda: self._track_goal(state))

        available_spawns = [
            sp
            for sp in spawn_points
            if sp.location.distance(ego_location) > 10.0
            and sp.location.distance(goal_spawn_location) >= 1.0
        ]
        self._rng.shuffle(available_spawns)
        target_npc_vehicles = max(0, int(self.config.num_npc_vehicles))
        spawned_npcs = 0
        for sp in available_spawns:
            if spawned_npcs >= target_npc_vehicles:
                break
            actor = runtime.actors.spawn_npc_vehicle(sp)
            if actor is not None:
                spawned_npcs += 1

        world = runtime.world.world
        spawned_pedestrians = 0
        occupied_locations = [ego_location]
        for actor in list(runtime.actors._actors):
            try:
                if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker.")):
                    occupied_locations.append(actor.get_location())
            except Exception:
                continue
        for _ in range(self.config.num_pedestrians):
            for _attempt in range(10):
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                loc.z += 0.5
                if any(loc.distance(other) < 5.0 for other in occupied_locations):
                    continue
                actor = runtime.actors.spawn_pedestrian(carla.Transform(loc))
                if actor is not None:
                    spawned_pedestrians += 1
                    occupied_locations.append(loc)
                    break

        logger.info(
            "Navigation actors spawned: vehicles=%d/%d pedestrians=%d/%d",
            spawned_npcs,
            self.config.num_npc_vehicles,
            spawned_pedestrians,
            self.config.num_pedestrians,
        )

        info = dict(state.get("info") or {})
        info.update(
            {
                "scenario_type": "navigation",
                "goal_location": goal_loc,
                "success_radius": self.config.success_radius,
                "initial_distance": initial_distance,
                "npc_vehicles_spawned": spawned_npcs,
                "pedestrians_spawned": spawned_pedestrians,
            }
        )
        state["info"] = info

    def _pick_goal(
        self,
        ego_location: carla.Location,
        spawn_points: list[carla.Transform],
        carla_map: carla.Map,
    ) -> Tuple[float, float, float]:
        if not self.config.random_goal and self.config.goal_location is not None:
            return self.config.goal_location

        from .._carla_agents.navigation.global_route_planner import GlobalRoutePlanner

        planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        candidates = list(spawn_points)
        self._rng.shuffle(candidates)
        reachable_fallback: carla.Transform | None = None
        reachable_fallback_rank: tuple[float, float] | None = None
        for sp in candidates:
            route_distance = float("inf")
            try:
                route = planner.trace_route(ego_location, sp.location)
            except Exception:
                route = None
            if not route:
                continue
            route_distance = _route_length_m(ego_location, sp.location, route)
            if self.config.route_distance_min <= route_distance <= self.config.route_distance_max:
                return (sp.location.x, sp.location.y, sp.location.z)
            if route_distance < self.config.route_distance_min:
                gap = self.config.route_distance_min - route_distance
            else:
                gap = route_distance - self.config.route_distance_max
            rank = (float(gap), float(route_distance))
            if reachable_fallback_rank is None or rank < reachable_fallback_rank:
                reachable_fallback = sp
                reachable_fallback_rank = rank

        if reachable_fallback is not None:
            logger.warning(
                "No reachable goal found within %.1f-%.1fm; using reachable fallback at %.1fm.",
                self.config.route_distance_min,
                self.config.route_distance_max,
                reachable_fallback_rank[1]
                if reachable_fallback_rank is not None
                else ego_location.distance(reachable_fallback.location),
            )
            return (
                reachable_fallback.location.x,
                reachable_fallback.location.y,
                reachable_fallback.location.z,
            )

        raise RuntimeError("NavigationScenario: could not find a reachable goal spawn")

    def _goal_distance(self, state: Any) -> float:
        runtime = state.get("carla")
        goal = state.get("scenario_data", {}).get("goal_location")
        if runtime is None or goal is None:
            return float("inf")
        ego_loc = runtime.ego_vehicle.get_location()
        goal_loc = _coerce_goal_location(goal)
        return float(goal_loc.distance(ego_loc))

    def _track_goal(self, state: Any) -> None:
        """Record the closest approach and goal arrival between tool calls."""
        navigation_state = state["scenario_state"]["navigation"]
        distance = self._goal_distance(state)
        best = navigation_state.get("best_distance_m")
        navigation_state["best_distance_m"] = distance if best is None else min(best, distance)
        if distance < float(self.config.success_radius):
            navigation_state["goal_reached"] = True

    def _goal_reached(self, state: Any) -> bool:
        navigation_state = state.get("scenario_state", {}).get("navigation", {})
        return bool(navigation_state.get("goal_reached")) or self._goal_distance(state) < float(
            self.config.success_radius
        )

    def is_done(self, state: Any) -> bool:
        if int(state.get("env_step", 0)) >= int(self.config.max_steps):
            return True
        if self._goal_reached(state):
            return True
        runtime = state.get("carla")
        if runtime is not None and runtime.collision_sensor.collision_count > 0:
            return True
        return False

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        """Score 0 after a collision, 1 at the goal, else the best progress toward it."""
        navigation_state = state.get("scenario_state", {}).get("navigation", {})
        goal_distance = self._goal_distance(state)
        initial_distance = float(navigation_state.get("initial_route_distance") or 1.0)
        best_distance = min(
            float(navigation_state.get("best_distance_m") or goal_distance), goal_distance
        )
        navigation_state["best_distance_m"] = best_distance
        progress = max(0.0, min(1.0, 1.0 - best_distance / max(initial_distance, 1.0)))

        runtime = state.get("carla")
        collision = bool(runtime is not None and runtime.collision_sensor.collision_count > 0)
        goal_reached = self._goal_reached(state)
        reward = 0.0 if collision else (1.0 if goal_reached else progress)

        outcome = {
            "scenario": self.config.name,
            "goal_reached": goal_reached,
            "goal_distance": float(goal_distance),
            "collision": collision,
            "progress": progress,
            "reward": float(reward),
            "route_distance_total": float(initial_distance),
            "route_distance_remaining": float(goal_distance),
        }
        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)
        return outcome

    def build_system_prompt(self, state: Any) -> str:
        goal = state.get("scenario_data", {}).get("goal_location")
        camera_available = bool(state.get("_camera_available", self.config.enable_vision))
        goal_info_available = bool(state.get("_goal_info_available", self.goal_info_enabled()))
        goal_line = ""
        if goal_info_available and goal is not None:
            goal_line = (
                f"Destination coordinates: x={goal[0]:.1f}, y={goal[1]:.1f}, z={goal[2]:.1f}\n"
            )
        if self.config.vision_only:
            vision_note = "Inspect the road through the available vision observations.\n"
            if not camera_available:
                vision_note = "Vision sensors unavailable in this episode.\n"
            objective_block = (
                f"Goal: navigate to within {self.config.success_radius:.0f}m of the destination.\n"
                f"{goal_line}"
                "You may query coarse goal progress, but not directional hints.\n"
                if goal_info_available
                else "This task does not use a live navigation goal.\n"
            )
            return (
                "Complete the vision-only navigation task.\n\n"
                f"{objective_block}"
                "You do not receive text observations about nearby actors, lanes, or goal distance.\n"
                f"{vision_note}"
            )
        camera_note = ""
        if self.config.enable_vision and not camera_available:
            camera_note = (
                "Front camera unavailable in this episode; rely on text observations only.\n"
            )
        objective_block = (
            f"Goal: navigate to within {self.config.success_radius:.0f}m of the destination.\n"
            "Avoid collisions with other vehicles and pedestrians.\n"
            f"{goal_line}"
            if goal_info_available
            else "Goal-based navigation is disabled for this task.\n"
        )
        return f"Complete the open navigation task.\n\n{objective_block}{camera_note}"

    def ticks_after_tool(self, tool_name: str, tool_args: dict, state: Any) -> int:
        if tool_name in {
            "capture_image",
            "get_goal_info",
            "init_navigation_agent",
            "set_destination",
            "follow_route",
        }:
            return 0
        return 0 if state.get("_tool_did_tick") else 1
