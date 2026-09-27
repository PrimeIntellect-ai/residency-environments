"""Procedural-family constants, prompts, and inaction scoring.

This module does not import the CARLA client, so the taskset worker builds prompts and
scores rollouts without tool calls from it; the scenarios read the same constants.
"""

from __future__ import annotations

from typing import Any

MAZE_SUCCESS_RADIUS_M = 12.0
NAVIGATION_SUCCESS_RADIUS_M = 10.0
COVERAGE_CELL_M = 20.0
COVERAGE_TARGET_CELLS = 30

COLLISION_RULE = "A collision ends the episode with a score of zero."


def procedural_prompt(family: str, modality: str) -> str:
    """Static task prompt for the maze, navigation, and free-roam families.

    The goal tool omits the compass direction in vision mode.
    """
    vision = modality == "vision"
    if family == "maze":
        reports = "its distance" if vision else "its distance and a coarse compass direction"
        return (
            f"Drive to a hidden destination and get within {MAZE_SUCCESS_RADIUS_M:.0f} m of it.\n\n"
            f"Its location is not revealed; get_goal_info reports {reports}."
        )
    if family == "navigation":
        reports = (
            "the destination coordinates and its distance"
            if vision
            else "the destination coordinates, its distance, and a coarse compass direction"
        )
        return (
            f"Drive to the destination and get within {NAVIGATION_SUCCESS_RADIUS_M:.0f} m of it.\n\n"
            f"get_goal_info reports {reports}. Other vehicles and pedestrians share the road. "
            f"{COLLISION_RULE}"
        )
    if family == "free_roam":
        return (
            f"Explore the town: drive through as many distinct {COVERAGE_CELL_M:.0f} m map "
            f"cells as you can. Reaching {COVERAGE_TARGET_CELLS} new cells earns the full "
            "score.\n\n"
            f"Other vehicles and pedestrians share the road. {COLLISION_RULE}"
        )
    raise ValueError(f"Unknown procedural family: {family}")


def procedural_inaction_outcome(scenario: str) -> dict[str, Any]:
    """A rollout without tool calls makes no progress and covers no new ground."""
    return {"scenario": scenario, "reward": 0.0, "episode_started": False}
