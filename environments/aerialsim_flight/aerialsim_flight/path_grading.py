"""Route rules evaluated on the piecewise-linear path between recorder samples.

Positions, box bounds and gate normals use the recorder's world NED coordinates.
Only ``ceiling_m`` is a height above the recorded resting ground level. Corridors
cover the entire recording, including launch and landing; there is no implicit
exemption before takeoff. Collision and final-state checks belong to the caller.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Point = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
EPS = 1e-9
PATH_METRICS = ("path_recording", "path_checkpoints", "path_gates", "path_ceiling", "path_corridor")


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Checkpoint(_Rule):
    """Enter a sphere. Starting inside counts once; a repeat needs a new entry."""

    kind: Literal["checkpoint"] = "checkpoint"
    name: str = Field(min_length=1)
    center: Point
    radius_m: float = Field(gt=0)


class Gate(_Rule):
    """Cross a circular aperture from its normal's negative to positive side.

    Use separately named gates for approach and re-entry, even when they share
    geometry. Starting on the plane, touching it and retreating, or hovering on
    it does not count as crossing it.
    """

    kind: Literal["gate"] = "gate"
    name: str = Field(min_length=1)
    center: Point
    normal: Point
    radius_m: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_normal(self) -> Self:
        length = math.hypot(*self.normal)
        if not math.isfinite(length) or length == 0:
            raise ValueError("a gate needs a finite, nonzero normal")
        return self


class Corridor(_Rule):
    """An allowed axis-aligned 3D box, including its boundary, in world NED."""

    minimum: Point
    maximum: Point

    @model_validator(mode="after")
    def positive_volume(self) -> Self:
        if any(lo >= hi for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("corridor minimum must be below maximum on every axis")
        return self


class PathRules(_Rule):
    """Ordered checkpoint/gate visits, plus whole-flight spatial constraints.

    An encounter before the preceding step is complete is not credited: the
    drone must visit that step again. One encounter cannot satisfy two steps.
    Empty corridors disable the corridor constraint; None disables the ceiling.
    """

    steps: tuple[Annotated[Checkpoint | Gate, Field(discriminator="kind")], ...] = ()
    ceiling_m: float | None = Field(default=None, ge=0)
    corridors: tuple[Corridor, ...] = ()

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        if len({step.name for step in self.steps}) != len(self.steps):
            raise ValueError("path step names must be unique")
        return self


def _sub(a: Point, b: Point) -> Point:
    return tuple(x - y for x, y in zip(a, b))


def _dot(a: Point, b: Point) -> float:
    return sum(x * y for x, y in zip(a, b))


def _checkpoint_hits(samples: list[tuple[float, Point]], checkpoint: Checkpoint) -> list[float]:
    center, radius = checkpoint.center, checkpoint.radius_m
    hits = [samples[0][0]] if math.dist(samples[0][1], center) <= radius + EPS else []
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        if math.dist(a, center) <= radius + EPS:
            continue  # Occupancy is not a new entry.
        delta, offset = _sub(b, a), _sub(a, center)
        aa = _dot(delta, delta)
        if aa == 0:
            continue
        bb = 2 * _dot(offset, delta)
        cc = _dot(offset, offset) - radius * radius
        discriminant = bb * bb - 4 * aa * cc
        if discriminant < -EPS:
            continue
        root = math.sqrt(max(0.0, discriminant))
        entry, leave = (-bb - root) / (2 * aa), (-bb + root) / (2 * aa)
        if leave >= 0 and -EPS <= entry <= 1 + EPS:
            u = min(1.0, max(0.0, entry))
            crossing = tuple(x + u * (y - x) for x, y in zip(a, b))
            # Check in metres too: a near-zero discriminant on a tiny segment
            # must not turn a distant miss into an apparent tangent contact.
            if math.dist(crossing, center) <= radius + EPS:
                hits.append(ta + u * (tb - ta))
    return hits


def _gate_hits(samples: list[tuple[float, Point]], gate: Gate) -> list[float]:
    length = math.hypot(*gate.normal)
    normal = tuple(v / length for v in gate.normal)

    def distance(point):
        return _dot(_sub(point, gate.center), normal)

    def side(value):
        return 1 if value > EPS else -1 if value < -EPS else 0

    last_side = side(distance(samples[0][1]))
    hits = []
    for (ta, a), (tb, b) in zip(samples, samples[1:]):
        da, db = distance(a), distance(b)
        next_side = side(db)
        if last_side == -1 and next_side == 1:
            # If a lies on the plane, we have evidence of the negative side
            # from an earlier sample. Starting on the plane alone is insufficient.
            u = -da / (db - da) if da < -EPS else 0.0
            crossing = tuple(x + u * (y - x) for x, y in zip(a, b))
            if math.dist(crossing, gate.center) <= gate.radius_m + EPS:
                hits.append(ta + u * (tb - ta))
        if next_side:
            last_side = next_side
    return hits


def _box_interval(a: Point, b: Point, box: Corridor) -> tuple[float, float] | None:
    """The portion of segment a->b inside a box, as parameters in [0, 1]."""
    enter, leave = 0.0, 1.0
    for start, end, lo, hi in zip(a, b, box.minimum, box.maximum):
        delta = end - start
        if abs(delta) <= EPS:
            if start < lo - EPS or start > hi + EPS:
                return None
            continue
        first, last = sorted(((lo - start) / delta, (hi - start) / delta))
        enter, leave = max(enter, first), min(leave, last)
        if enter > leave + EPS:
            return None
    return enter, leave


def _inside_corridors(a: Point, b: Point, corridors: tuple[Corridor, ...]) -> bool:
    intervals = sorted(interval for box in corridors if (interval := _box_interval(a, b, box)) is not None)
    covered = 0.0
    for enter, leave in intervals:
        if enter > covered + EPS:
            return False
        covered = max(covered, leave)
        if covered >= 1.0 - EPS:
            return True
    return False


def grade_path(
    recording: list[dict], rules: PathRules, ground_z: float, max_sample_gap_s: float = 1.0
) -> dict[str, float]:
    """Return binary route metrics; callers require every metric to pass.

    Interpolation assumes straight travel between adjacent samples. Invalid or
    gappy telemetry is rejected rather than used to invent crossings. This does
    not replace the caller's takeoff, collision or independent endpoint checks.
    """
    if not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0:
        raise ValueError("max_sample_gap_s must be finite and positive")
    failed = dict.fromkeys(PATH_METRICS, 0.0)
    try:
        samples = [(float(row["t"]), tuple(float(row[axis]) for axis in ("x", "y", "z"))) for row in recording]
        if not math.isfinite(ground_z) or len(samples) < 2:
            return failed
        if any(not all(math.isfinite(v) for v in (t, *point)) or t < 0 for t, point in samples):
            return failed
        if any(not 0 < tb - ta <= max_sample_gap_s + EPS for (ta, _), (tb, _) in zip(samples, samples[1:])):
            return failed
    except (KeyError, TypeError, ValueError, OverflowError):
        return failed

    completed = []
    cursor = -math.inf
    for step in rules.steps:
        hits = _checkpoint_hits(samples, step) if isinstance(step, Checkpoint) else _gate_hits(samples, step)
        hit = next((t for t in hits if t > cursor + EPS), None)
        if hit is None:
            break
        completed.append(step.name)
        cursor = hit

    return {
        "path_recording": 1.0,
        "path_checkpoints": float(all(step.name in completed for step in rules.steps if isinstance(step, Checkpoint))),
        "path_gates": float(all(step.name in completed for step in rules.steps if isinstance(step, Gate))),
        "path_ceiling": float(
            rules.ceiling_m is None or all(ground_z - p[2] <= rules.ceiling_m + EPS for _, p in samples)
        ),
        "path_corridor": float(
            not rules.corridors
            or all(_inside_corridors(a, b, rules.corridors) for (_, a), (_, b) in zip(samples, samples[1:]))
        ),
    }
