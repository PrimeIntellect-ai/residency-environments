from __future__ import annotations

from typing import TYPE_CHECKING

from ..decisions import TrolleyAction, classify_trolley_action

if TYPE_CHECKING:
    import carla

__all__ = ["TrolleyAction", "classify_trolley_action", "same_direction"]


def same_direction(a: carla.Waypoint, b: carla.Waypoint) -> bool:
    """True if two waypoints face the same direction (same-sign lane_id)."""
    try:
        return (a.lane_id * b.lane_id) > 0
    except Exception:
        return False
