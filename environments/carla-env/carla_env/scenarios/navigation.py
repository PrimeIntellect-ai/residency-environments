from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import carla

from ..logging import get_logger
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
    success_radius: float = 10.0
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
        state.setdefault("scenario_state", {})
        state["scenario_state"]["navigation"] = {
            "prev_goal_distance": None,
            "initial_route_distance": None,
            "best_distance_m": None,
            "collision_count": 0,
            "cumulative_reward": 0.0,
        }
        if self._configured_weather == "random":
            self.config.weather = random.choice(WEATHER_PRESETS)

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
        navigation_state["prev_goal_distance"] = initial_distance
        navigation_state["best_distance_m"] = initial_distance

        available_spawns = [
            sp
            for sp in spawn_points
            if sp.location.distance(ego_location) > 10.0
            and sp.location.distance(goal_spawn_location) >= 1.0
        ]
        random.shuffle(available_spawns)
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
        random.shuffle(candidates)
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

    def is_done(self, state: Any) -> bool:
        if int(state.get("env_step", 0)) >= int(self.config.max_steps):
            return True
        if self._goal_distance(state) < float(self.config.success_radius):
            return True
        runtime = state.get("carla")
        if runtime is not None and runtime.collision_sensor.collision_count > 0:
            return True
        return False

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        navigation_state = state.get("scenario_state", {}).get("navigation", {})
        goal_distance = self._goal_distance(state)
        initial_distance = float(navigation_state.get("initial_route_distance") or 1.0)
        previous_distance = float(navigation_state.get("prev_goal_distance") or goal_distance)

        runtime = state.get("carla")
        collision = bool(runtime is not None and runtime.collision_sensor.collision_count > 0)
        progress = (previous_distance - goal_distance) / max(initial_distance, 1.0)
        goal_reached = goal_distance < float(self.config.success_radius)
        step_reward = (
            progress + (10.0 if goal_reached else 0.0) + (-5.0 if collision else 0.0) - 0.01
        )
        cumulative_reward = float(navigation_state.get("cumulative_reward") or 0.0) + step_reward

        navigation_state["prev_goal_distance"] = goal_distance
        navigation_state["cumulative_reward"] = cumulative_reward

        outcome = {
            "scenario": self.config.name,
            "goal_reached": goal_reached,
            "goal_distance": float(goal_distance),
            "collision": collision,
            "reward": float(cumulative_reward),
            "step_reward": float(step_reward),
            "route_distance_total": float(initial_distance),
            "route_distance_remaining": float(goal_distance),
        }
        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)
        return outcome

    def build_system_prompt(self, state: Any) -> str:
        goal = state.get("scenario_data", {}).get("goal_location")
        camera_available = bool(state.get("_camera_available", self.config.enable_vision))
        depth_available = bool(state.get("_depth_available", False))
        observe_available = bool(state.get("_observe_available", self.observe_tool_enabled()))
        goal_info_available = bool(state.get("_goal_info_available", self.goal_info_enabled()))
        motion_tools_enabled = self.motion_tools_enabled()
        replay_note = self.replay_readonly_note()
        goal_line = ""
        if goal_info_available and goal is not None:
            goal_line = (
                f"Destination coordinates: x={goal[0]:.1f}, y={goal[1]:.1f}, z={goal[2]:.1f}\n"
            )
        if self.config.vision_only:
            vision_note = (
                "Use the available vision tools to inspect the road scene before taking actions.\n"
            )
            vision_tools = ""
            if camera_available:
                vision_tools += "- capture_image(): capture the current front RGB camera image\n"
            if not camera_available and depth_available:
                vision_note = "RGB camera unavailable in this episode; use depth capture to inspect the road scene.\n"
            elif not camera_available and not depth_available:
                vision_note = "Vision sensors unavailable in this episode.\n"
            if depth_available:
                vision_tools += "- capture_depth(): capture the current front depth camera image\n"
            objective_block = (
                f"Goal: navigate to within {self.config.success_radius:.0f}m of the destination.\n"
                f"{goal_line}"
                "You may query coarse goal progress, but not directional hints.\n"
                if goal_info_available
                else "This replay does not use a live navigation goal.\n"
            )
            goal_tools = (
                "- get_goal_info(): get distance/progress to goal without direction\n"
                if goal_info_available
                else ""
            )
            return (
                "You control a vehicle in a vision-only navigation environment.\n\n"
                f"{objective_block}"
                "You do not receive text observations about nearby actors, lanes, or goal distance.\n"
                f"{replay_note}"
                f"{vision_note}"
                "Available tools:\n"
                f"{vision_tools}"
                + (
                    "- observe(): advance the replay without receiving text observations\n"
                    if observe_available
                    else ""
                )
                + goal_tools
                + (
                    "- control_vehicle(throttle, steer): manual control\n"
                    "- brake_vehicle(intensity): apply brakes\n"
                    "- lane_change(direction): change lane left/right\n"
                    "- init_navigation_agent(behavior): start autopilot\n"
                    "- set_destination(x, y, z): set navigation goal\n"
                    "- follow_route(steps): follow planned route\n"
                    "- emergency_stop(): stop immediately\n"
                    if motion_tools_enabled
                    else ""
                )
            )
        camera_note = ""
        if self.config.enable_vision and not camera_available:
            if depth_available:
                camera_note = "Front RGB camera unavailable in this episode; depth capture is still available.\n"
            else:
                camera_note = (
                    "Front camera unavailable in this episode; rely on text observations only.\n"
                )
        objective_block = (
            f"Goal: navigate to within {self.config.success_radius:.0f}m of the destination.\n"
            "Avoid collisions with other vehicles and pedestrians.\n"
            f"{goal_line}"
            if goal_info_available
            else "Goal-based navigation is disabled for this replay.\n"
        )
        tools = "Available tools:\n"
        if goal_info_available:
            tools += "- get_goal_info(): distance and direction to goal\n"
        if observe_available:
            tools += "- observe(): get current state\n"
        if motion_tools_enabled:
            tools += (
                "- control_vehicle(throttle, steer): manual control\n"
                "- brake_vehicle(intensity): apply brakes\n"
                "- lane_change(direction): change lane left/right\n"
                "- init_navigation_agent(behavior): start autopilot\n"
                "- set_destination(x, y, z): set navigation goal\n"
                "- follow_route(steps): follow planned route\n"
                "- emergency_stop(): stop immediately\n"
            )
        if camera_available:
            tools += "- capture_image(): capture the front RGB camera\n"
        if depth_available:
            tools += "- capture_depth(): capture the front depth camera\n"
        strategy_note = (
            "Recommended strategy: initialize a navigation agent, set the destination, "
            "then follow the route while observing regularly.\n\n"
            if motion_tools_enabled
            else (
                "Recommended strategy: inspect the replay as it unfolds and use goal progress queries when needed.\n\n"
                if goal_info_available
                else "Recommended strategy: inspect the replay as it unfolds.\n\n"
            )
        )
        return (
            "You control a vehicle in an open navigation environment.\n\n"
            f"{objective_block}"
            f"{replay_note}"
            f"{camera_note}"
            f"{strategy_note}"
            f"{tools}"
        )

    def ticks_after_tool(self, tool_name: str, tool_args: dict, state: Any) -> int:
        if tool_name in {"capture_image", "capture_depth", "get_goal_info", "follow_route"}:
            return 0
        if state.get("_nurec_drive") and tool_name in {
            "control_vehicle",
            "brake_vehicle",
            "emergency_stop",
        }:
            return max(4, int(round(float(self.config.nurec_framerate or 20.0) * 0.25)))
        return 0 if state.get("_tool_did_tick") else 1
