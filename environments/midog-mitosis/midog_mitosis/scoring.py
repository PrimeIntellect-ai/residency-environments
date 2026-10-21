"""Answer parsing and scoring for midog-mitosis. Pure functions with no verifiers dependency.

A reply must end with ``Answer: [[x, y], ...]`` at the start of a line; markdown emphasis (``*``, ``_``, backticks) is
ignored. The last non-empty ``Answer:`` is the answer, and everything after it must be that one JSON list of ``[x, y]``
number pairs (``[]`` is an explicit "none"). Coordinates are
normalized to 0-1000 over the image width and height. The reward is binary: 1 if the listed points and the annotated
mitotic figures match one-to-one within ``MATCH_RADIUS_PX`` native pixels, none missing and none extra; 0 otherwise,
including any answer that cannot be read unambiguously.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

POINT_SCALE = 1000.0
MATCH_RADIUS_PX = 30.0
"""A predicted point matches a mitotic figure when it lies within this many native pixels of its center: 7.5 um at
0.25 um per pixel, the detection criterion of the MIDOG challenges."""

_ANSWER = re.compile(
    r"^[ \t>#-]*(?:final[ \t]+)?answer[ \t]*:[ \t]*(.*?)(?=^[ \t>#-]*(?:final[ \t]+)?answer[ \t]*:|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class Evaluation:
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)


def answer_tail(reply: str) -> str | None:
    """The text after the last non-empty ``Answer:`` that starts a line, with ``*``, ``_`` and backticks removed."""
    tails = [t.strip() for t in _ANSWER.findall(re.sub(r"[*_`]", "", reply))]
    non_empty = [t for t in tails if t]
    return non_empty[-1] if non_empty else None


def parse_points(reply: str) -> list[tuple[float, float]] | None:
    """The ``[x, y]`` pairs of the answer, or None when the answer is missing or is not exactly one list of number pairs."""
    tail = answer_tail(reply)
    if tail is None:
        return None
    try:
        points = json.loads(tail)
    except ValueError:
        return None
    if not isinstance(points, list) or not all(
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in point)
        for point in points
    ):
        return None
    return [(float(x), float(y)) for x, y in points]


def max_matching(predicted: list[tuple[float, float]], gold: list[tuple[float, float]], radius: float) -> int:
    """Size of the largest one-to-one matching in which every matched pair lies within ``radius`` (augmenting paths)."""
    near = [[j for j, g in enumerate(gold) if math.dist(p, g) <= radius] for p in predicted]
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in near[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def score_find(reply: str, gold_px: list[tuple[float, float]], width: int, height: int) -> Evaluation:
    """Reward is 1 when the predicted centers and the annotated centers match one-to-one, else 0.

    ``gold_px`` are pixel coordinates in an image of ``width`` x ``height``; the reply is normalized to 0-1000 on each axis.
    The metrics (F1, precision, recall) say how close a failed answer was; they never enter the reward.
    """
    predicted = parse_points(reply)
    if predicted is None:
        return Evaluation(
            0.0, {"exact": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "answered": 0.0, "predicted": 0.0}
        )
    scaled = [(x * width / POINT_SCALE, y * height / POINT_SCALE) for x, y in predicted]
    tp = max_matching(scaled, gold_px, MATCH_RADIUS_PX)
    fp, fn = len(scaled) - tp, len(gold_px) - tp
    exact = float(tp == len(gold_px) and fp == 0)
    metrics = {
        "exact": exact,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "answered": 1.0,
        "predicted": float(len(scaled)),
    }
    return Evaluation(exact, metrics)
