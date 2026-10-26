"""Seeded visual decisions, public routes and replay checks for families 4–7.

Scene facts stay private. Every conditional route is explained in the same
mission text for all variants of a seed. This module never calls a simulator.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aerialsim_flight.path_grading import (
    EPS,
    Checkpoint,
    Corridor,
    Gate,
    PathRules,
    _checkpoint_hits,
    _gate_hits,
)
from aerialsim_flight.timed_events import EventSchedule, TimedEvent

ADVANCED_FAMILIES = ("alternate_pad", "patrol_inspection", "go_around", "compound")
VARIANT_COUNTS = dict(zip(ADVANCED_FAMILIES, (3, 2, 2, 4)))
EVALUATION_BASES = dict(zip(ADVANCED_FAMILIES, (3_000_000, 4_000_000, 5_000_000, 6_000_000)))
SEEDS_PER_SPLIT = 10_000
PAD_RADIUS_M = 2.5
HOLD_RADIUS_M = 1.0
HOLD_S = 3.0
PAD_A_NAME = "pad_marker"
PAD_B_NAME = "alternate_pad_marker"
BLOCK_A_NAME = "post_on_pad"
BLOCK_B_NAME = "post_on_alternate_pad"
CUE_NAME = "inspection_cue"
BARRIER_NAME = "compound_barrier"


class AdvancedScenario(BaseModel):
    """A complete private scenario identity, with deterministic derived geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    family: Literal["alternate_pad", "patrol_inspection", "go_around", "compound"]
    seed: int = Field(ge=0)
    split: Literal["development", "evaluation"]
    variant: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_identity(self):
        if self.variant >= VARIANT_COUNTS[self.family]:
            raise ValueError("variant is outside this family's complete counterfactual group")
        base = EVALUATION_BASES[self.family] if self.split == "evaluation" else 0
        if not base <= self.seed < base + SEEDS_PER_SPLIT:
            raise ValueError("seed must belong to this family's reserved split")
        return self

    @property
    def pad_x(self) -> float:
        low, high = (22.0, 22.4) if self.split == "development" else (23.0, 23.4)
        return round(random.Random(f"advanced-layout-v1:{self.family}:{self.seed}").uniform(low, high), 3)

    @property
    def clear_after(self) -> float:
        low, high = (30.0, 40.0) if self.split == "development" else (42.0, 48.0)
        return round(random.Random(f"advanced-event-v1:{self.family}:{self.seed}").uniform(low, high), 2)


def make_scenario(family: str, seed: int, split: str, variant: int) -> AdvancedScenario:
    return AdvancedScenario(family=family, seed=seed, split=split, variant=variant)


def turn_limit(family: str) -> int:
    if family not in ADVANCED_FAMILIES:
        raise ValueError("unknown advanced family")
    return 48 if family in ("go_around", "compound") else 28


class PublicGeometry(BaseModel):
    """Only the coordinates already disclosed in the public mission text.

    The rehearsal harness validates this before handing anything to a blind
    control. No cue, occupancy, seed, event timing or correct ending belongs here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    pad_a: tuple[float, float]
    pad_b: tuple[float, float]
    hold: tuple[float, float] = (16.0, 0.0)
    inspection: tuple[float, float, float] = (6.0, 0.0, -2.0)
    merge: tuple[float, float, float] = (17.0, 0.0, -2.0)
    branch_positive: tuple[float, float, float] = (14.0, 4.0, -3.0)
    branch_negative: tuple[float, float, float] = (14.0, -4.0, -3.0)
    bypass_entry: tuple[float, float, float] = (8.0, 4.25, -2.0)
    bypass: tuple[float, float, float] = (14.0, 4.25, -2.0)
    approach: tuple[float, float, float] = (20.0, 3.5, -2.0)
    retreat: tuple[float, float, float] = (16.0, 3.5, -4.0)
    climb: tuple[float, float, float] = (16.0, 0.0, -4.0)
    reentry_stage: tuple[float, float, float] = (17.0, 3.5, -2.0)

    @model_validator(mode="after")
    def fixed_public_geometry(self):
        x = self.pad_a[0]
        if not math.isfinite(x) or not (22.0 <= x <= 22.4 or 23.0 <= x <= 23.4):
            raise ValueError("pad X must lie in a reserved public layout range")
        if self.pad_a != (x, 3.5) or self.pad_b != (x, -3.5):
            raise ValueError("public pad coordinates must use the declared A/B geometry")
        for name, field in type(self).model_fields.items():
            if name not in ("pad_a", "pad_b") and getattr(self, name) != field.default:
                raise ValueError(f"{name} must use the declared public geometry")
        return self


def public_geometry(scenario: AdvancedScenario) -> dict:
    return PublicGeometry(pad_a=(scenario.pad_x, 3.5), pad_b=(scenario.pad_x, -3.5)).model_dump(mode="json")


def _checkpoint(name: str, point: tuple, radius: float = 1.25) -> Checkpoint:
    return Checkpoint(name=name, center=point, radius_m=radius)


def _approach(name="approach") -> Gate:
    return Gate(name=name, center=(19, 3.5, -2), normal=(1, 0, 0), radius_m=1.0)


def _breakoff() -> Gate:
    return Gate(name="breakoff", center=(17, 3.5, -4), normal=(-1, 0, 0), radius_m=1.0)


def needs_go_around(scenario: AdvancedScenario) -> bool:
    return scenario.family in ("go_around", "compound") and scenario.variant == 1


def expected_ending(scenario: AdvancedScenario) -> Literal["land", "abort"]:
    abort = (scenario.family == "alternate_pad" and scenario.variant == 2) or (
        scenario.family == "compound" and scenario.variant == 3
    )
    return "abort" if abort else "land"


def expected_target(scenario: AdvancedScenario) -> tuple[float, float]:
    if expected_ending(scenario) == "abort":
        return (16.0, 0.0)
    use_b = (scenario.family in ("alternate_pad", "patrol_inspection") and scenario.variant == 1) or (
        scenario.family == "compound" and scenario.variant == 2
    )
    return (scenario.pad_x, -3.5 if use_b else 3.5)


def route_rules(scenario: AdvancedScenario) -> PathRules:
    steps: list[Checkpoint | Gate] = [_checkpoint("inspection", (6, 0, -2))]
    if scenario.family == "patrol_inspection":
        steps.append(_checkpoint("remembered_branch", (14, 4 if scenario.variant == 0 else -4, -3), 0.75))
    if scenario.family == "compound":
        steps.append(_checkpoint("bypass", (14, 4.25, -2), 1.0))
    steps.append(_checkpoint("merge", (17, 0, -2)))
    if scenario.family == "go_around" or (scenario.family == "compound" and scenario.variant in (0, 1)):
        steps.append(_approach())
        if needs_go_around(scenario):
            steps.extend((_breakoff(), _checkpoint("climb", (16, 0, -4), 0.6), _approach("reentry")))
    corridors = (Corridor(minimum=(-2, -6, -30), maximum=(26, 6, 3)),)
    if scenario.family == "compound":
        corridors = (
            Corridor(minimum=(-2, -6, -30), maximum=(8, 6, 3)),
            Corridor(minimum=(8, 3, -30), maximum=(14, 5.5, 3)),
            Corridor(minimum=(14, -6, -30), maximum=(26, 6, 3)),
        )
    return PathRules(steps=tuple(steps), ceiling_m=7, corridors=corridors)


def _object(name: str, x: float, y: float, sx=1.0, sy=1.0, sz=2.0, asset="Cube", colour=None) -> dict:
    out = dict(name=name, asset=asset, x=x, y=y, z=0.0, sx=sx, sy=sy, sz=sz)
    if colour is not None:
        out["colour"] = colour
    return out


def scene_objects(scenario: AdvancedScenario) -> list[dict]:
    objects = [
        _object(PAD_A_NAME, scenario.pad_x, 3.5, 3, 3, 0.02, "Cylinder", [35, 35, 40]),
        _object(PAD_B_NAME, scenario.pad_x, -3.5, 3, 3, 0.02, "Cylinder", [35, 35, 40]),
    ]
    if scenario.family == "alternate_pad":
        if scenario.variant in (1, 2):
            objects.append(_object(BLOCK_A_NAME, scenario.pad_x, 3.5))
        if scenario.variant == 2:
            objects.append(_object(BLOCK_B_NAME, scenario.pad_x, -3.5))
    elif scenario.family == "patrol_inspection":
        objects.append(_object(CUE_NAME, 6, 2 if scenario.variant == 0 else -2, 0.8, 0.8, 1.5))
    elif scenario.family == "compound":
        objects.append(_object(BARRIER_NAME, 11.5, 0, 1.5, 4, 8))
        if scenario.variant in (2, 3):
            objects.append(_object(BLOCK_A_NAME, scenario.pad_x, 3.5))
        if scenario.variant == 3:
            objects.append(_object(BLOCK_B_NAME, scenario.pad_x, -3.5))
    if needs_go_around(scenario):
        # These are outside the pad and approach; their visible pairing means
        # temporary closure, independently of the hidden clearance time.
        objects.extend(
            (
                _object("temporary_left", scenario.pad_x - 1, 1.3, 0.4, 0.4, 0.8),
                _object("temporary_right", scenario.pad_x - 1, 5.7, 0.4, 0.4, 0.8),
            )
        )
    return objects


def event_schedule(scenario: AdvancedScenario) -> EventSchedule | None:
    if needs_go_around(scenario):
        obj = _object(BLOCK_A_NAME, scenario.pad_x, 3.5)
        obj.pop("name")
        obj.pop("z")
        return EventSchedule(
            trigger_x=17.0,
            events=(
                TimedEvent(at_s=0.0, op="spawn", name=BLOCK_A_NAME, object=obj),
                TimedEvent(at_s=scenario.clear_after, op="destroy", name=BLOCK_A_NAME),
            ),
        )
    if scenario.family == "patrol_inspection":
        return EventSchedule(trigger_x=10.0, events=(TimedEvent(at_s=0.0, op="destroy", name=CUE_NAME),))
    return None


def mission_text(scenario: AdvancedScenario) -> str:
    common = (
        f"Mission: pad A is at ({scenario.pad_x:g}, 3.5), pad B at ({scenario.pad_x:g}, -3.5). "
        "Use the camera to inspect their dark round markers. White blocks on a pad mean blocked. "
        "All coordinates below are world NED metres. Remain within x=-2..26 and y=-6..6, "
        "and at or below 7 m above ground. Fly through inspection (6, 0, -2) first, "
        "within 1.25 m, and complete the specified route before ending. "
        "Land within 2.5 m horizontally of the chosen pad, with one final touchdown only; "
        "do not land and take off again. If instructed to abort, first fly to the safe hold "
        "point (16, 0), then abort: remain within 1 m horizontally of that point, "
        "5 m above ground, for at least 3 seconds. "
    )
    if scenario.family == "alternate_pad":
        return common + (
            "Then reach merge (17, 0, -2), within 1.25 m. A is preferred: land on A if clear, "
            "otherwise land on B if clear. If neither is clear, abort at the safe hold point. "
            "A and B can both be clear; in that case use A. "
            "How to fly it: take off, go to inspection (6, 0, -2), then straight to merge "
            "(17, 0, -2). From merge look with pitch -30 and yaw 0: both pads are in view "
            "ahead, A on the right of the frame and B on the left. A white block on a dark "
            "marker means that pad is blocked; a bare dark marker is clear. Judge both pads "
            "from that one look, then fly to the chosen pad at z=-2, look with pitch -80 to "
            "confirm it is bare, and land. If neither is clear go to (16, 0, -2) and abort. "
            "Do not fly over the pads to inspect them first."
        )
    if scenario.family == "patrol_inspection":
        return common + (
            "At inspection, look down for the single small white cue beside (6, 0): if it is on "
            "the positive-Y side, remember positive; if on the negative-Y side, remember negative. "
            "The cue disappears once you pass x=10. Next enter the remembered branch checkpoint "
            "(14, 4, -3) for positive or (14, -4, -3) for negative, within 0.75 m. "
            "Do not enter the other branch checkpoint. Then reach merge (17, 0, -2), within "
            "1.25 m. Both pads are clear: positive means land on A, negative means land on B."
        )
    if scenario.family == "compound":
        common += (
            "A tall white wall blocks the direct route after inspection. While x is strictly "
            "between 8 and 14, stay in the corridor y=3..5.5; use staging (8, 4.25, -2), "
            "then bypass (14, 4.25, -2) within 1 m, then merge (17, 0, -2) within 1.25 m. "
        )
    else:
        # The merge waypoint has to read as part of every flight, not as a step of the
        # abort procedure that ends `common`. Stated straight after that sentence, a model
        # skips it and cuts the corner from inspection to the approach gate, missing merge
        # against a 1.25 m tolerance and failing path_checkpoints on every flight, which
        # makes the mission unreachable. The other families escape it only because they go
        # on to state their route in full, so do not shorten them without checking.
        common += (
            "The shared route is inspection then merge: after inspection fly straight to "
            "merge (17, 0, -2), within 1.25 m. Fly it on every mission, not only when "
            "aborting. "
        )
    common += (
        "Inspect A again during approach. Two small white posts flanking A indicate a temporary "
        "closure: its block appears as you pass x=17 and will clear later. Its clearance time "
        "is not given; use the camera to see it clear. For A, line up via (17, 3.5, -2), then cross the approach gate at "
        "(19, 3.5, -2) travelling toward increasing X, within 1 m of its centre. "
        "When temporary closure is indicated, make this first approach while A is blocked, "
        "stop short of the pad near (20, 3.5, -2), then immediately make the missed approach: "
        "climb and retreat through (17, 3.5, -4) toward decreasing X within 1 m before "
        "the block clears, then reach "
        "climb checkpoint (16, 0, -4) within 0.6 m. Wait safely there if needed and look again. "
        "After the block clears, return via (17, 3.5, -2) and cross the same approach gate "
        "again toward increasing X, then land on A. Waiting in place without this maneuver "
        "does not count. When A is clear and has no temporary-closure posts, approach and land "
        "directly; do not perform the missed approach or visit its climb checkpoint. "
    )
    if scenario.family == "compound":
        common += (
            "If A has a block but no temporary-closure posts, it is permanently blocked: "
            "land on B if B is clear, otherwise abort at the safe hold point. Inspect both "
            "pads after completing the shared route. A temporary closure requires the missed "
            "approach to A even if B is clear."
        )
    else:
        common += "Only A is the landing target in this mission; B is not an alternate."
    return common


def _finite(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _near(state: Mapping | None, point: tuple, radius: float) -> bool:
    if not isinstance(state, Mapping):
        return False
    axes = ("x", "y", "z")[: len(point)]
    values = tuple(state.get(axis) for axis in axes)
    return all(_finite(v) for v in values) and math.dist(values, point) <= radius + EPS


def _accepted_actions(history: list[dict] | None):
    for entry in history or ():
        if isinstance(entry, Mapping) and isinstance(action := entry.get("action"), Mapping):
            yield action


def _terminal_choice(scenario: AdvancedScenario, decision: str | None, state: Mapping | None) -> str | None:
    if decision == "land":
        if _near(state, (scenario.pad_x, 3.5), PAD_RADIUS_M):
            return "land_a"
        if _near(state, (scenario.pad_x, -3.5), PAD_RADIUS_M):
            return "divert_b"
        return "other"
    if decision == "abort":
        return "hold" if _near(state, (16, 0), HOLD_RADIUS_M) else "other"
    return None


def _ending_choice(
    scenario: AdvancedScenario, history: list[dict] | None, decision: str | None, state: Mapping | None
) -> str | None:
    """Freeze the first accepted command that distinguishes the four endings.

    Approach staging is deliberately not a landing commitment: (20, 3.5) is
    already within the final landing tolerance on some layouts. A goto must
    target the pad centre within 0.75 m to commit to landing. An accepted retreat
    or climb from the approach counts before its controller result, so a later
    stall cannot turn a correct visual choice into an incorrect decision.
    """
    for entry in history or ():
        if not isinstance(entry, Mapping) or not isinstance(action := entry.get("action"), Mapping):
            continue
        before = entry.get("state")
        if action.get("do") == "goto":
            if _near(action, (scenario.pad_x, 3.5), 0.75):
                return "land_a"
            if _near(action, (scenario.pad_x, -3.5), 0.75):
                return "divert_b"
            if _near(before, (20, 3.5, -2), 1.5) and (
                _near(action, (16, 3.5, -4), 1.0) or _near(action, (16, 0, -4), 0.6)
            ):
                return "go_around"
            # Any climbing retreat off the approach is a go around decision, whatever
            # the exact retreat point: the text names one, the grader does not require it.
            if (
                isinstance(before, Mapping)
                and _finite(before.get("x"))
                and _finite(action.get("x"))
                and _finite(action.get("z"))
                and before["x"] >= 18.0
                and action["x"] <= before["x"] - 1.0
                and action["z"] <= -3.5
            ):
                return "go_around"
        elif (choice := _terminal_choice(scenario, action.get("do"), before)) is not None:
            return choice
    # The caller retains accepted terminal state separately; support its use
    # when replaying a decision without an action-history trace.
    return _terminal_choice(scenario, decision, state)


def _branch_choice(history: list[dict] | None) -> int | None:
    # The first accepted command to one of the publicly named branch checkpoints
    # is a decision even if execution fails. Later correction does not erase it.
    for action in _accepted_actions(history):
        if action.get("do") == "goto":
            for variant, side in ((0, 4), (1, -4)):
                if _near(action, (14, side, -3), 0.75):
                    return variant
    return None


def decision_score(
    scenario: AdvancedScenario, decision: str | None, state: dict | None, action_history: list[dict] | None = None
) -> float:
    if scenario.family == "patrol_inspection":
        # Report the remembered branch choice independently of later landing.
        return float(_branch_choice(action_history) == scenario.variant)
    if scenario.family in ("go_around", "compound"):
        expected = ("land_a", "go_around", "divert_b", "hold")[scenario.variant]
        return float(_ending_choice(scenario, action_history, decision, state) == expected)
    # Alternate pad: the choice is the first pad the model heads for, or the hold,
    # whether or not the flight then reaches it. Execution is graded elsewhere.
    expected = ("land_a", "divert_b", "hold")[scenario.variant]
    return float(_ending_choice(scenario, action_history, decision, state) == expected)


def scenario_obstacle_hit(recording: list[dict] | None) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("hit") in (BLOCK_A_NAME, BLOCK_B_NAME, BARRIER_NAME)
        and _finite(row.get("collision_t"))
        and row["collision_t"] > 0
        for row in recording or ()
    )


def _samples(recording: list[dict], clock="t") -> list[tuple[float, tuple]] | None:
    try:
        samples = [(r[clock], (r["x"], r["y"], r["z"])) for r in recording]
        if len(samples) < 2 or not all(_finite(t) and t >= 0 and all(_finite(v) for v in p) for t, p in samples):
            return None
        if any(not 0 <= b[0] - a[0] <= 1.0 + EPS for a, b in zip(samples, samples[1:])):
            return None
        return samples
    except (KeyError, TypeError, ValueError):
        return None


def _event(log: dict | None, name: str, op: str) -> dict | None:
    if not isinstance(log, Mapping):
        return None
    if not isinstance(log.get("events"), list):
        return None
    entries = [
        e
        for e in log["events"]
        if isinstance(e, Mapping) and e.get("name") == name and e.get("op") == op and e.get("status") == "ok"
    ]
    if len(entries) != 1:
        return None
    event = entries[0]
    if not all(_finite(event.get(key)) and event[key] >= 0 for key in ("started_sim_ns", "fired_sim_ns")):
        return None
    if event["fired_sim_ns"] < event["started_sim_ns"]:
        return None
    return event


def _one_final_touchdown(recording: list[dict], rest_z: float) -> bool:
    airborne = False
    touched = False
    for row in recording:
        height = rest_z - row["z"]
        if height > 0.5:
            if touched:
                return False
            airborne = True
        # After proven flight, a ground-level excursion counts even if the
        # simulator's landed flag lags. The reset itself predates airborne.
        if airborne and height <= 0.3:
            touched = True
    return airborne


def _ordered(*hit_lists: list[float]) -> bool:
    cursor = -math.inf
    for hits in hit_lists:
        cursor = next((value for value in hits if value > cursor + EPS), math.inf)
        if not math.isfinite(cursor):
            return False
    return True


def extra_grade(
    scenario: AdvancedScenario,
    recording: list[dict],
    final: dict | None,
    event_log: dict | None = None,
    action_history: list[dict] | None = None,
    rest_z: float = 0.0,
) -> dict[str, float]:
    """Additional binary gates; shared route/collision/touchdown checks are separate.

    Event times are absolute simulator seconds. Wall-clock delays while a model
    thinks cannot satisfy clearance or a timed hold. Failed/missing event logs
    fail the temporal contract even if the final position happens to be right.
    """
    out = dict.fromkeys(
        ("sim_recording", "single_touchdown", "inspection_branch", "missed_approach", "event_timing", "hold_position"),
        1.0,
    )
    samples = _samples(recording)
    if samples is None or not _finite(rest_z):
        return dict.fromkeys(out, 0.0)
    sim = _samples(recording, "sim_t")
    out["sim_recording"] = float(sim is not None and sim[-1][0] > sim[0][0])
    out["single_touchdown"] = float(_one_final_touchdown(recording, rest_z))
    if scenario.family == "patrol_inspection":
        wrong = _checkpoint("wrong", (14, -4 if scenario.variant == 0 else 4, -3), 0.75)
        out["inspection_branch"] = float(not _checkpoint_hits(samples, wrong))
        removed = _event(event_log, CUE_NAME, "destroy")
        branch = next(step for step in route_rules(scenario).steps if step.name == "remembered_branch")
        inspection = route_rules(scenario).steps[0]
        seen = _checkpoint_hits(sim, inspection) if sim else []
        reached = _checkpoint_hits(sim, branch) if sim else []
        out["event_timing"] = float(
            bool(removed and seen and reached)
            and seen[0] < removed["started_sim_ns"] / 1e9 <= removed["fired_sim_ns"] / 1e9 < reached[0]
        )
    if scenario.family in ("go_around", "compound") and scenario.variant in (0, 1):
        approach = _gate_hits(samples, _approach())
        retreat = _gate_hits(samples, _breakoff())
        climb = _checkpoint_hits(samples, _checkpoint("climb", (16, 0, -4), 0.6))
        if not needs_go_around(scenario):
            out["missed_approach"] = float(not retreat and not climb)
        else:
            # Every stage needs a distinct later encounter; holding on a gate
            # or reusing a single approach crossing cannot satisfy re-entry.
            out["missed_approach"] = float(_ordered(approach, retreat, climb, approach))
            spawned = _event(event_log, BLOCK_A_NAME, "spawn")
            removed = _event(event_log, BLOCK_A_NAME, "destroy")
            hits = _gate_hits(sim, _approach()) if sim else []
            retreats = _gate_hits(sim, _breakoff()) if sim else []
            climbs = _checkpoint_hits(sim, _checkpoint("climb", (16, 0, -4), 0.6)) if sim else []
            # Timing and ordered maneuver must describe the SAME sequence.
            out["event_timing"] = float(
                bool(spawned and removed)
                and _ordered(
                    [a for a in hits if spawned["fired_sim_ns"] / 1e9 <= a < removed["started_sim_ns"] / 1e9],
                    [b for b in retreats if b < removed["started_sim_ns"] / 1e9],
                    climbs,
                    [d for d in hits if d >= removed["fired_sim_ns"] / 1e9],
                )
            )
    if expected_ending(scenario) == "abort":
        if sim is None:
            out["hold_position"] = 0.0
        else:
            end = sim[-1][0]
            tail = [(t, p) for t, p in sim if t >= end - HOLD_S - EPS]
            out["hold_position"] = float(
                len(tail) >= 3
                and tail[-1][0] - tail[0][0] >= HOLD_S - 0.15
                and all(
                    math.dist(p[:2], expected_target(scenario)) <= HOLD_RADIUS_M + EPS
                    and abs(rest_z - p[2] - 5.0) <= 1.5
                    for _, p in tail
                )
            )
    return out


__all__ = [
    "ADVANCED_FAMILIES",
    "VARIANT_COUNTS",
    "EVALUATION_BASES",
    "SEEDS_PER_SPLIT",
    "AdvancedScenario",
    "PublicGeometry",
    "make_scenario",
    "public_geometry",
    "turn_limit",
    "mission_text",
    "scene_objects",
    "route_rules",
    "event_schedule",
    "expected_target",
    "expected_ending",
    "decision_score",
    "extra_grade",
    "scenario_obstacle_hit",
    "needs_go_around",
]
