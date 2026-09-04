"""Helper for AnomalyXL multi-channel templates: choose K non-overlapping
anomaly windows across K distinct channels.

Used by `localize_all_channels` (K ∈ {0..3} per row, channels picked uniformly
from the pool). Windows on *different* channels can overlap in time — that's not
a collision for set-matching purposes (the scorer requires both channel and
window to match).
"""

from __future__ import annotations

import numpy as np

from ..anomalies import ANOMALY_KINDS, InjectionSpec, make_injection
from ..difficulty import Difficulty

__all__ = ["plan_k_anomalies"]


def plan_k_anomalies(
    names: list[str],
    length: int,
    k: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 32,
    difficulty: "Difficulty | None" = None,
) -> list[InjectionSpec]:
    """Choose `k` anomaly windows across `k` distinct channels, building nothing.

    `k = 0` is a first-class case and returns `[]`.
    """
    difficulty = difficulty or Difficulty()
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return []
    if k > len(names):
        raise ValueError(f"k={k} > n_channels={len(names)} with distinct-channel constraint")

    idx = rng.choice(len(names), size=k, replace=False)
    placed: list[InjectionSpec] = []
    per_channel_windows: dict[str, list[tuple[int, int]]] = {}

    for channel in (names[int(i)] for i in idx):
        kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        for _ in range(max_attempts):
            candidate = make_injection(
                length,
                rng,
                kind=kind,
                channel=channel,
                min_width=difficulty.width_range[0],
                max_width=difficulty.width_range[1],
                magnitude_range=difficulty.magnitude_range,
            )
            if not _overlaps_any(candidate.t_start, candidate.t_end, per_channel_windows.get(channel, [])):
                break
        else:
            # The window distribution is wide; hitting the cap means the channel
            # is saturated rather than unlucky.
            raise RuntimeError(
                f"Could not place anomaly on channel {channel!r} within {max_attempts} attempts (k={k}, L={length})."
            )
        per_channel_windows.setdefault(channel, []).append((candidate.t_start, candidate.t_end))
        placed.append(candidate)

    return placed


def _overlaps_any(start: int, end: int, windows: list[tuple[int, int]]) -> bool:
    """Half-open interval overlap check against a list of existing windows."""
    for s, e in windows:
        if start < e and s < end:
            return True
    return False
