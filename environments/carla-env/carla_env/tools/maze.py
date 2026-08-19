from __future__ import annotations

import math
from typing import Any

import carla


def _runtime(state: Any):
    rt = state.get("carla")
    if rt is None:
        raise RuntimeError("CARLA runtime not initialized")
    return rt


def get_goal_info(state: Any = None) -> str:
    """
    Report distance + rough direction to the goal.
    """
    if state is None:
        return "Error: no state"

    rt = _runtime(state)
    goal_loc: carla.Location | None = None
    tracking_state: dict | None = None
    maze_state = state.get("scenario_state", {}).get("maze", {})
    maze_goal = maze_state.get("goal")
    if isinstance(maze_goal, dict):
        try:
            goal_loc = carla.Location(
                x=float(maze_goal["x"]),
                y=float(maze_goal["y"]),
                z=float(maze_goal.get("z", 0.0)),
            )
            tracking_state = maze_state
        except Exception:
            pass

    if goal_loc is None:
        navigation_goal = state.get("scenario_data", {}).get("goal_location")
        if navigation_goal is not None:
            try:
                goal_loc = carla.Location(
                    x=float(navigation_goal[0]),
                    y=float(navigation_goal[1]),
                    z=float(navigation_goal[2]),
                )
                tracking_state = state.get("scenario_state", {}).get("navigation", {})
            except Exception:
                pass

    if goal_loc is None:
        return "Error: goal not set"

    ego_loc = rt.ego_vehicle.get_location()
    dx = goal_loc.x - ego_loc.x
    dy = goal_loc.y - ego_loc.y
    dist = float(goal_loc.distance(ego_loc))

    # Direction in world XY (not heading-relative): N/S/E/W.
    angle = math.degrees(math.atan2(dy, dx))
    # CARLA world: +x east-ish, +y north-ish depending on map; we keep generic.
    if -45 <= angle < 45:
        cardinal = "E"
    elif 45 <= angle < 135:
        cardinal = "N"
    elif angle >= 135 or angle < -135:
        cardinal = "W"
    else:
        cardinal = "S"

    # Update best distance for progress shaping (pure state, no CARLA calls later).
    if tracking_state is None:
        improving = False
    else:
        best = tracking_state.get("best_distance_m")
        try:
            best_f = float(best) if best is not None else dist
        except Exception:
            best_f = dist
        if dist < best_f:
            tracking_state["best_distance_m"] = dist
            improving = True
        else:
            improving = False
    if bool(state.get("_vision_only", False)):
        return f"distance_to_goal_m={dist:.1f} improving={improving}"
    return f"distance_to_goal_m={dist:.1f} direction={cardinal} improving={improving}"
