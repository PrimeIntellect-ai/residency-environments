"""Public route rules and private obstacle facts for the bypass family.

Both scene variants share exactly the same route. Variant 0 leaves the positive
Y lane open; variant 1 leaves the negative Y lane open. The wall spans from
1 m past the centre line on the open side to 12 m on the blocked side. Positions are world NED,
including the checkpoint Z values. Only the ceiling is relative to resting
ground height, as in the shared path grader.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from aerialsim_flight.path_grading import EPS, Checkpoint, Corridor, PathRules

BARRIER_NAME = "route_barrier"
CHOICE_X = 8.0
RouteChoice = Literal["positive", "negative", "centre", "outside"]


def route_rules() -> PathRules:
    """The public route, independent of obstacle location and open side."""
    return PathRules(
        steps=(
            Checkpoint(name="inspection", center=(6.0, 0.0, -2.0), radius_m=1.5),
            Checkpoint(name="merge", center=(18.0, 0.0, -2.0), radius_m=1.5),
        ),
        ceiling_m=6.0,
        corridors=(
            Corridor(minimum=(-2.0, -5.5, -30.0), maximum=(8.0, 5.5, 3.0)),
            Corridor(minimum=(8.0, 3.0, -30.0), maximum=(16.0, 5.5, 3.0)),
            Corridor(minimum=(8.0, -5.5, -30.0), maximum=(16.0, -3.0, 3.0)),
            Corridor(minimum=(16.0, -5.5, -30.0), maximum=(24.0, 5.5, 3.0)),
        ),
    )


def _valid_variant(variant: int) -> None:
    if type(variant) is not int or variant not in (0, 1):
        raise ValueError("obstacle variant must be 0 or 1")


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def barrier_object(obstacle_x: float, variant: int) -> dict:
    """A static white wall; the caller sets Z from ground height before spawn."""
    _valid_variant(variant)
    if not _finite_number(obstacle_x) or not 10.0 <= obstacle_x <= 14.0:
        raise ValueError("obstacle x must be finite and between 10 and 14")
    # The wall runs from one metre past the centre line on the open side out to
    # 12 m on the blocked side. Looking straight ahead from the inspection
    # checkpoint, one half of the view is wall and the other half open grass,
    # at any wall distance. A wall centred 2 m off the line showed grass past
    # its far end on both sides; one centred 5 m off still needed two angled
    # looks that stopped telling the sides apart at the evaluation distances.
    return {
        "name": BARRIER_NAME,
        "asset": "Cube",
        "x": float(obstacle_x),
        "y": -5.5 if variant == 0 else 5.5,
        "z": 0.0,
        "sx": 1.5,
        "sy": 13.0,
        "sz": 8.0,
    }


def route_choice(state: Mapping[str, Any], action: Mapping[str, Any]) -> RouteChoice | None:
    """Classify the first accepted forward goto entering the passage at X>8.

    The caller must retain the first non-None result before executing that action;
    later corrections do not replace the initial choice. Staging on X=8 does
    not commit a choice, but even a short move beyond it does. For a start before
    the passage, interpolate at X=8; for a start already inside, use its current
    Y. A destination in a side lane cannot excuse entry through the centre.
    """
    if action.get("do") != "goto":
        return None
    values = (state.get("x"), state.get("y"), action.get("x"), action.get("y"))
    if not all(_finite_number(value) for value in values):
        return None
    x, y, target_x, target_y = values
    if target_x <= x or target_x <= CHOICE_X or x >= 16.0:
        return None
    fraction = max(0.0, (CHOICE_X - x) / (target_x - x))
    entry_y = y + fraction * (target_y - y)
    if 3.0 - EPS <= entry_y <= 5.5 + EPS:
        return "positive"
    if -5.5 - EPS <= entry_y <= -3.0 + EPS:
        return "negative"
    return "outside" if abs(entry_y) > 5.5 + EPS else "centre"


def choice_score(choice: RouteChoice | None, variant: int) -> float:
    """Whether the first commanded passage entry chooses the open lane."""
    _valid_variant(variant)
    return float(choice == ("positive" if variant == 0 else "negative"))


def barrier_hit(recording: list[Any] | None) -> bool:
    """Whether trusted telemetry contains a timestamped hit on this wall.

    This is diagnostic evidence, not a substitute for the complete collision and
    mission checks. Match the exact spawned name so a ground or pad collision
    cannot make a blind pilot's expected wall-collision check pass.
    """
    return any(
        isinstance(row, Mapping)
        and row.get("hit") == BARRIER_NAME
        and _finite_number(stamp := row.get("collision_t"))
        and stamp > 0.0
        for row in (recording or ())
    )


__all__ = [
    "BARRIER_NAME",
    "CHOICE_X",
    "RouteChoice",
    "route_rules",
    "barrier_object",
    "route_choice",
    "choice_score",
    "barrier_hit",
]
