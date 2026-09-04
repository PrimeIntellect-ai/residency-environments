"""Orchestration: choose a sample, then build it.

`plan_sample` decides what a sample contains and derives its question and gold
without touching an array; `materialize` turns that plan into channel data. The
three seeds are independent, so changing scene generation cannot move an anomaly.
"""

from __future__ import annotations

import numpy as np

from .difficulty import Difficulty
from .questions import TEMPLATES, Row, SamplePlan

__all__ = ["Row", "SamplePlan", "materialize", "plan_sample"]


def plan_sample(
    category: str,
    length: int,
    n_channels: int,
    *,
    plan_seed: int,
    scene_seed: int,
    injection_seed: int,
    paired_split: tuple[float, float] | None = None,
    difficulty: Difficulty | None = None,
) -> SamplePlan:
    if category not in TEMPLATES:
        raise KeyError(f"Unknown category {category!r}; available: {sorted(TEMPLATES)}")
    return TEMPLATES[category]().plan(
        length,
        n_channels,
        np.random.default_rng(plan_seed),
        scene_seed=scene_seed,
        injection_seed=injection_seed,
        paired_split=paired_split,
        difficulty=difficulty,
    )


def materialize(plan: SamplePlan) -> Row:
    return TEMPLATES[plan.category]().materialize(plan)
