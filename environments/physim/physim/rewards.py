"""Reward shaping applied once, after averaging scientific energy scores."""

from __future__ import annotations

import math

DEFAULT_PRECISION = 2.0
REWARD_MAPPING = "max(0, min(-log10(S), K)) / K; S=0 -> 1; invalid predictor -> 0"


def precision_label(precision: float) -> str:
    """Human-readable, round-trip-safe target for prompts and run identity."""
    return str(float(precision)).removesuffix(".0")


def precision_reward(energy: float | None, precision: float = DEFAULT_PRECISION) -> float:
    """Map normalized energy to a bounded reward; K is a positive log10 target.

    None denotes a missing/invalid predictor, not a zero-energy forecast.
    Negative or nonfinite energies indicate an upstream scoring error.
    """
    if not math.isfinite(precision) or precision <= 0:
        raise ValueError("reward precision K must be positive and finite")
    if energy is None:
        return 0.0
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("energy must be nonnegative and finite, or None for an invalid predictor")
    if energy == 0:
        return 1.0
    return max(0.0, min(-math.log10(energy), precision)) / precision
