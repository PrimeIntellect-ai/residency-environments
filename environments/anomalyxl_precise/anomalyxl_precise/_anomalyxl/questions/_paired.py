"""Shared helpers for Tier III paired templates (`LeadLagWithMagnitude`).

Both templates need the same plumbing: split `n_channels` into `(C_A, C_B)`,
build two scenes, sample a `CouplingSpec`, inject anomalies as the coupling
dictates, and merge the channel dicts. Only the question text + option set +
gold-derivation differ.
"""

from __future__ import annotations

import numpy as np

from ..anomalies import (
    ANOMALY_KINDS,
    MAX_WIDTH,
    MIN_WIDTH,
    CouplingSpec,
    InjectionSpec,
)
from ..difficulty import Difficulty

__all__ = ["split_channels", "plan_for_coupling"]


def split_channels(total: int, paired_split: tuple[float, float] | None) -> tuple[int, int]:
    """`(C_A, C_B)` summing to `total`. Default split is balanced; the larger
    side gets the leftover when `total` is odd."""
    if total < 2:
        raise ValueError(f"Paired templates need n_channels >= 2, got {total}")
    if paired_split is None:
        a = total // 2
        return a, total - a
    sa, sb = paired_split
    if sa <= 0 or sb <= 0:
        raise ValueError(f"paired_split components must be positive, got {paired_split}")
    a = max(1, int(round(total * sa / (sa + sb))))
    a = min(a, total - 1)
    return a, total - a


def _sample_window(
    L: int,
    rng: np.random.Generator,
    *,
    must_fit_after: int = 0,
    width_range: tuple[int, int] = (MIN_WIDTH, MAX_WIDTH),
) -> tuple[int, int, int]:
    """`(t_start, t_end, width)` with `t_start >= must_fit_after` and window in `[0, L)`."""
    max_w = min(width_range[1], L - must_fit_after)
    if max_w < width_range[0]:
        raise ValueError(f"Cannot fit min-width window after position {must_fit_after} in L={L}")
    width = int(rng.integers(width_range[0], max_w + 1))
    latest_start = L - width
    if latest_start < must_fit_after:
        latest_start = must_fit_after
    t_start = int(rng.integers(must_fit_after, latest_start + 1))
    return t_start, t_start + width, width


def _pick_target(names: list[str], rng: np.random.Generator) -> str:
    return names[int(rng.integers(0, len(names)))]


def plan_for_coupling(
    names_a: list[str],
    names_b: list[str],
    length: int,
    coupling: CouplingSpec,
    rng: np.random.Generator,
    *,
    difficulty: "Difficulty | None" = None,
) -> tuple[list[InjectionSpec], CouplingSpec]:
    """Choose the anomalies the coupling calls for, building nothing.

    Returns the specs and the (possibly re-clamped) `CouplingSpec`. One anomaly
    per side at most.
    """
    difficulty = difficulty or Difficulty()
    kind = coupling.kind
    L = length
    injections: list[InjectionSpec] = []

    if kind == "no_anomaly":
        return [], coupling

    if kind in {"one_sided_A", "one_sided_B"}:
        target = _pick_target(names_a if kind == "one_sided_A" else names_b, rng)
        anomaly_kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        t_start, t_end, _ = _sample_window(L, rng, width_range=difficulty.width_range)
        injections.append(
            InjectionSpec(
                channel=target,
                t_start=t_start,
                t_end=t_end,
                kind=anomaly_kind,
                magnitude=float(rng.uniform(*difficulty.magnitude_range)),
            )
        )
        return injections, coupling

    if kind == "independent":
        # "Independent" must be distinguishable from the coupled classes by the
        # documented rule (coupled offsets never exceed lag_max_fraction * L), so
        # enforce a start-to-start separation strictly greater than that bound —
        # and give either series the earlier window equally often, so temporal
        # order carries no signal about the class.
        anomaly_kind_a = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        anomaly_kind_b = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
        min_gap = int(L * difficulty.lag_max_fraction) + 1
        w_early = int(rng.integers(difficulty.width_range[0], min(difficulty.width_range[1], L) + 1))
        w_late = int(rng.integers(difficulty.width_range[0], min(difficulty.width_range[1], L) + 1))
        max_early_start = L - w_late - min_gap
        if max_early_start < 0:
            raise ValueError(f"L={L} too short for independent pair with min_gap={min_gap}")
        t_early = int(rng.integers(0, max_early_start + 1))
        t_late = int(rng.integers(t_early + min_gap, L - w_late + 1))
        if rng.integers(0, 2) == 0:
            t_start_a, t_end_a = t_early, t_early + w_early
            t_start_b, t_end_b = t_late, t_late + w_late
        else:
            t_start_a, t_end_a = t_late, t_late + w_late
            t_start_b, t_end_b = t_early, t_early + w_early
        target_a = _pick_target(names_a, rng)
        target_b = _pick_target(names_b, rng)
        injections.extend(
            [
                InjectionSpec(
                    channel=target_a,
                    t_start=t_start_a,
                    t_end=t_end_a,
                    kind=anomaly_kind_a,
                    magnitude=float(rng.uniform(*difficulty.magnitude_range)),
                ),
                InjectionSpec(
                    channel=target_b,
                    t_start=t_start_b,
                    t_end=t_end_b,
                    kind=anomaly_kind_b,
                    magnitude=float(rng.uniform(*difficulty.magnitude_range)),
                ),
            ]
        )
        return injections, coupling

    # Coupled cases: same kind, same width; B offset by `coupling.lag`.
    anomaly_kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
    lag = coupling.lag if kind != "coupled_simul" else 0

    # Pick A's window such that B's start = A's start + lag lands in `[0, L - width]`.
    max_w = min(difficulty.width_range[1], L - abs(lag))
    if max_w < difficulty.width_range[0]:
        # Too short to express the lag; degrade to simultaneous.
        lag = 0
        coupling = CouplingSpec(kind="coupled_simul", lag=0)
        max_w = min(difficulty.width_range[1], L)
    width = int(rng.integers(difficulty.width_range[0], max_w + 1))
    a_low = max(0, -lag)
    a_high = min(L - width, L - width - lag)
    if a_high < a_low:
        a_high = a_low
    t_start_a = int(rng.integers(a_low, a_high + 1))
    t_start_b = t_start_a + lag
    target_a = _pick_target(names_a, rng)
    target_b = _pick_target(names_b, rng)
    magnitude = float(rng.uniform(*difficulty.magnitude_range))
    injections.extend(
        [
            InjectionSpec(
                channel=target_a,
                t_start=t_start_a,
                t_end=t_start_a + width,
                kind=anomaly_kind,
                magnitude=magnitude,
            ),
            InjectionSpec(
                channel=target_b,
                t_start=t_start_b,
                t_end=t_start_b + width,
                kind=anomaly_kind,
                magnitude=magnitude,
            ),
        ]
    )
    return injections, coupling
