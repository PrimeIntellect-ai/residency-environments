from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import carla

from ..logging import get_logger
from .base import BaseScenario, ScenarioConfig

logger = get_logger("scenarios.maze")


@dataclass
class MazeConfig(ScenarioConfig):
    min_goal_distance_m: float = 80.0
    max_goal_distance_m: float = 300.0
    success_radius_m: float = 12.0
    seed: Optional[int] = None


class MazeScenario(BaseScenario[MazeConfig]):
    """
    Navigation scenario: reach a hidden goal location.

    Goal is selected from map spawn points. Reward is progress-based (0..1),
    with 1.0 for reaching the target within the success radius.
    """

    def __init__(self, config: MazeConfig):
        super().__init__(config)
        self._rng = random.Random()

    def supports_goal_info(self) -> bool:
        return True

    def build_system_prompt(self, state: Any) -> str:
        cfg = self.config
        replay_note = self.replay_readonly_note()
        goal_info_available = bool(state.get("_goal_info_available", self.goal_info_enabled()))
        objective_block = (
            f"Goal: reach the hidden target location (within {cfg.success_radius_m:.0f}m).\n"
            if goal_info_available
            else "This replay does not use an active hidden goal.\n"
        )
        goal_tools = (
            "- get_goal_info(): distance/direction to goal (use often)\n"
            if goal_info_available
            else ""
        )
        intro = (
            "Use tools to inspect the run and to query goal distance.\n\n"
            if goal_info_available
            else "Use tools to inspect the run as it unfolds.\n\n"
        )
        motion_tools = (
            "- init_navigation_agent(behavior): setup a CARLA agent\n"
            "- set_destination(x, y): pick a destination\n"
            "- follow_route(steps): let the agent drive for N ticks\n"
            "- lane_change(direction)\n"
            "- emergency_stop(), brake_vehicle(intensity), control_vehicle(throttle, steer)\n"
            if self.motion_tools_enabled()
            else ""
        )
        return (
            "You control a vehicle in a navigation task.\n\n"
            f"{objective_block}"
            f"{replay_note}"
            f"{intro}"
            "Available tools:\n"
            "- observe(): get a new observation\n"
            f"{goal_tools}"
            f"{motion_tools}"
        )

    def reset(self, state: Any) -> None:
        cfg = self.config
        self._rng.seed(cfg.seed)
        state.setdefault("scenario_state", {})
        state["scenario_state"]["maze"] = {
            "goal": None,
            "initial_distance_m": None,
            "best_distance_m": None,
            "progress": 0.0,
        }

    def setup(self, state: Any) -> None:
        runtime = state["carla"]
        cfg = self.config

        carla_map = runtime.world.map
        spawns = list(carla_map.get_spawn_points())
        if not spawns:
            raise RuntimeError("MazeScenario: map has no spawn points")

        ego_loc = runtime.ego_vehicle.get_location()

        # Pick a goal far enough from ego.
        candidates: list[carla.Location] = []
        for _ in range(200):
            loc = self._rng.choice(spawns).location
            d = float(loc.distance(ego_loc))
            if cfg.min_goal_distance_m <= d <= cfg.max_goal_distance_m:
                candidates.append(loc)

        goal = candidates[0] if candidates else self._rng.choice(spawns).location

        st = state["scenario_state"]["maze"]
        st["goal"] = {"x": float(goal.x), "y": float(goal.y), "z": float(goal.z)}
        d0 = float(goal.distance(ego_loc))
        st["initial_distance_m"] = d0
        st["best_distance_m"] = d0
        st["progress"] = 0.0

        info = state.get("info") if isinstance(state.get("info"), dict) else {}
        info = dict(info or {})
        info.update(
            {
                "scenario_type": "maze",
                "goal": st["goal"],
                "success_radius_m": float(cfg.success_radius_m),
                "initial_distance_m": float(d0),
            }
        )
        state["info"] = info

        logger.info("Maze goal selected: (%.1f, %.1f) initial_distance=%.1fm", goal.x, goal.y, d0)

    def _current_goal_location(self, state: Any) -> Optional[carla.Location]:
        st = state.get("scenario_state", {}).get("maze", {})
        goal = st.get("goal")
        if not isinstance(goal, dict):
            return None
        try:
            return carla.Location(
                x=float(goal["x"]), y=float(goal["y"]), z=float(goal.get("z", 0.0))
            )
        except Exception:
            return None

    def _update_progress(self, state: Any) -> None:
        runtime = state["carla"]
        st = state["scenario_state"]["maze"]
        goal = self._current_goal_location(state)
        if goal is None:
            return
        ego_loc = runtime.ego_vehicle.get_location()
        dist = float(goal.distance(ego_loc))

        best = float(st.get("best_distance_m") or dist)
        best = min(best, dist)
        st["best_distance_m"] = best

        d0 = st.get("initial_distance_m")
        if isinstance(d0, (int, float)) and d0 > 0:
            progress = max(0.0, min(1.0, (float(d0) - best) / float(d0)))
        else:
            progress = 0.0
        st["progress"] = progress

    def is_done(self, state: Any) -> bool:
        cfg = self.config
        self._update_progress(state)

        goal = self._current_goal_location(state)
        if goal is None:
            return True
        runtime = state["carla"]
        dist = float(goal.distance(runtime.ego_vehicle.get_location()))
        if dist <= float(cfg.success_radius_m):
            return True

        return int(state.get("env_step", 0)) >= int(cfg.max_steps)

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        cfg = self.config
        self._update_progress(state)
        st = state["scenario_state"]["maze"]

        goal = self._current_goal_location(state)
        runtime = state["carla"]
        dist = (
            float(goal.distance(runtime.ego_vehicle.get_location()))
            if goal is not None
            else float("inf")
        )
        reached = dist <= float(cfg.success_radius_m)

        progress = float(st.get("progress", 0.0) or 0.0)
        reward = progress if not reached else 1.0

        outcome = {
            "scenario": cfg.name,
            "distance_to_goal_m": dist,
            "progress": progress,
            "reached_goal": bool(reached),
            "reward": float(reward),
        }

        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)
        return outcome

    def ticks_after_tool(self, tool_name: str, tool_args: dict, state: Any) -> int:
        # Minimal advancement per tool; follow_route ticks internally.
        if tool_name == "follow_route":
            return 0
        return 1
