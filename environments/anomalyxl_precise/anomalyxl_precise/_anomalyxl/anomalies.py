"""Five anomaly transforms + `InjectionSpec` / `CouplingSpec` constructors.

The library design invariant: anomaly width bounds are **absolute index
counts**, not fractions of `L`. The ~256-sample max footprint is what makes
the long-context evaluation interesting (a full-series plot at `L=32k` looks
like noise unless you zoom in).

Each anomaly transform takes the channel's *current* values and a
`InjectionSpec` describing the window and the kind, and returns a fresh
array with the transform applied to `[t_start, t_end)`. No transform clips
to the channel range — magnitudes are expressed in units of `channel_std`,
which the caller computes from the pre-injection series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ANOMALY_KINDS",
    "COUPLING_KINDS",
    "MIN_WIDTH",
    "MAX_WIDTH",
    "MAX_COUPLING_LAG",
    "CouplingSpec",
    "InjectionSpec",
    "inject",
    "make_injection",
    "make_coupling",
]


ANOMALY_KINDS: tuple[str, ...] = (
    "level_shift",
    "transient_spike",
    "change_in_seasonality",
    "change_in_variance",
    "change_in_trend",
)

COUPLING_KINDS: tuple[str, ...] = (
    "independent",
    "coupled_simul",
    "coupled_lag_pos",
    "coupled_lag_neg",
    "one_sided_A",
    "one_sided_B",
    "no_anomaly",
)

# Absolute index bounds; do NOT make L-relative. See module docstring.
MIN_WIDTH: int = 8
MAX_WIDTH: int = 256
MAX_COUPLING_LAG: int = 512


@dataclass(frozen=True)
class InjectionSpec:
    channel: str
    t_start: int
    t_end: int
    kind: str
    magnitude: float  # in units of pre-injection channel std

    @property
    def width(self) -> int:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class CouplingSpec:
    """Tier III A↔B relationship descriptor.

    `lag` is `t_b_start - t_a_start` in indices (positive => B follows A).
    For non-coupled kinds (`independent`, `one_sided_*`, `no_anomaly`) `lag`
    is 0 by convention.
    """

    kind: str
    lag: int


def _channel_std(series: np.ndarray) -> float:
    s = float(series.std())
    # Constant channels: pretend std=1 so magnitudes have a defined scale.
    return s if s > 1e-9 else 1.0


def _level_shift(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    out = series.copy()
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    out[spec.t_start : spec.t_end] += sign * spec.magnitude * _channel_std(series)
    return out


def _transient_spike(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    out = series.copy()
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    width = spec.width
    # Triangular envelope: zero at edges, peak `magnitude * std` at the center.
    t = np.arange(width)
    envelope = 1.0 - np.abs((t - (width - 1) / 2.0) / max(1.0, (width - 1) / 2.0))
    out[spec.t_start : spec.t_end] += sign * spec.magnitude * _channel_std(series) * envelope
    return out


def _change_in_seasonality(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    """Add a *new* periodic signal over the window — the local seasonality
    changes character even if the base scene was aperiodic. Period is short
    enough to be resolved inside the window."""
    out = series.copy()
    width = spec.width
    # Period in [4, width // 2] so at least 2 cycles fit inside the window.
    period = float(rng.integers(4, max(5, width // 2 + 1)))
    amp = spec.magnitude * _channel_std(series)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(width)
    out[spec.t_start : spec.t_end] += amp * np.sin(2.0 * np.pi * t / period + phase)
    return out


def _change_in_variance(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    """Replace local values with `original + extra_noise(std = magnitude * channel_std)`
    so the marginal variance inside the window jumps by `(1 + magnitude^2)`."""
    out = series.copy()
    extra_std = spec.magnitude * _channel_std(series)
    out[spec.t_start : spec.t_end] += rng.normal(0.0, extra_std, size=spec.width)
    return out


def _change_in_trend(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    """Add a piecewise linear ramp over the window (zero at the start so the
    pre-window level is preserved, then drifting up or down). After the
    window, the ramp's terminal value carries forward — otherwise it would
    look like a transient pulse, not a trend change."""
    out = series.copy()
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    peak = sign * spec.magnitude * _channel_std(series)
    width = spec.width
    ramp = np.linspace(0.0, peak, width)
    out[spec.t_start : spec.t_end] += ramp
    if spec.t_end < out.shape[0]:
        out[spec.t_end :] += peak
    return out


_DISPATCH = {
    "level_shift": _level_shift,
    "transient_spike": _transient_spike,
    "change_in_seasonality": _change_in_seasonality,
    "change_in_variance": _change_in_variance,
    "change_in_trend": _change_in_trend,
}


def inject(series: np.ndarray, spec: InjectionSpec, rng: np.random.Generator) -> np.ndarray:
    """Apply `spec.kind` to `series[t_start:t_end]` and return a new array.

    The input is never mutated. Window bounds are clamped to `[0, len(series)]`
    by the caller (`make_injection` guarantees this); a transform that escapes
    the window indicates a bug, not a data condition to handle.
    """
    try:
        fn = _DISPATCH[spec.kind]
    except KeyError as e:
        raise KeyError(f"Unknown anomaly kind {spec.kind!r}; expected one of {ANOMALY_KINDS}") from e
    return fn(series, spec, rng)


def make_injection(
    L: int,
    rng: np.random.Generator,
    *,
    kind: str | None = None,
    channel: str = "",
    min_width: int = MIN_WIDTH,
    max_width: int = MAX_WIDTH,
    magnitude_range: tuple[float, float] = (2.0, 6.0),
    allow_runoff: bool = False,
) -> InjectionSpec:
    """Sample an `InjectionSpec` constrained to `[0, L)`.

    `kind=None` picks uniformly from `ANOMALY_KINDS`. `magnitude_range` is in
    units of `channel_std`. `allow_runoff=True` lets `t_end == L` (used by
    the End-Time "Not Resolved" template).
    """
    if L < min_width:
        raise ValueError(f"L={L} smaller than min_width={min_width}")
    effective_max = min(max_width, L)
    if effective_max < min_width:
        raise ValueError(f"max_width={max_width} clamps below min_width={min_width} for L={L}")
    chosen = kind or ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
    # `change_in_variance` adds Gaussian noise sample-by-sample, so a window
    # the size of MIN_WIDTH=8 with magnitude up to 6σ is visually indistinguishable
    # from a pair of transient spikes. Bump its floor so the *sustained* nature
    # of the variance change is what the model sees.
    kind_min_width = max(min_width, 64) if chosen == "change_in_variance" else min_width
    kind_min_width = min(kind_min_width, effective_max)
    width = int(rng.integers(kind_min_width, effective_max + 1))
    # Latest start so the window fits in [0, L]; the upper bound is exclusive.
    latest_start = L - width
    t_start = int(rng.integers(0, latest_start + 1))
    t_end = t_start + width
    if allow_runoff:
        # Re-bias toward t_end == L: with probability 0.5, slide window so it
        # ends exactly at L.
        if rng.uniform() < 0.5:
            t_start = latest_start
            t_end = L
    magnitude = float(rng.uniform(*magnitude_range))
    return InjectionSpec(
        channel=channel,
        t_start=t_start,
        t_end=t_end,
        kind=chosen,
        magnitude=magnitude,
    )


def make_coupling(
    L: int,
    rng: np.random.Generator,
    *,
    kind: str,
    max_lag: int = MAX_COUPLING_LAG,
) -> CouplingSpec:
    """Sample a `CouplingSpec`. `lag` is signed for `coupled_lag_*`; zero for
    other kinds. For `L < 4 * max_lag` the lag is clamped to `L // 4` so the
    paired anomalies stay distinct and inside the series."""
    if kind not in COUPLING_KINDS:
        raise KeyError(f"Unknown coupling kind {kind!r}; expected one of {COUPLING_KINDS}")
    effective_max = max(1, min(max_lag, L // 4))
    if kind == "coupled_lag_pos":
        lag = int(rng.integers(MIN_WIDTH, effective_max + 1))
    elif kind == "coupled_lag_neg":
        lag = -int(rng.integers(MIN_WIDTH, effective_max + 1))
    else:
        lag = 0
    return CouplingSpec(kind=kind, lag=lag)
