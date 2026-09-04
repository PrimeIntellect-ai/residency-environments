"""Base time-series primitives.

Every function takes an `rng: np.random.Generator` and returns a `(L,)`
`float64` array. **No randomness escapes the rng** — pass a child rng if you
want isolation.

Series are returned with their natural scale (no normalization here); scaling
to unit-ish variance is the scene assembly's job.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ar1",
    "constant",
    "exp_drift",
    "linear_drift",
    "log_drift",
    "red_noise",
    "sawtooth",
    "sine",
    "square",
    "white_noise",
    "BASE_NOISE_KINDS",
    "PERIODIC_KINDS",
    "TREND_KINDS",
    "draw_kind",
]


BASE_NOISE_KINDS: tuple[str, ...] = ("white_noise", "red_noise", "ar1")
PERIODIC_KINDS: tuple[str, ...] = ("sine", "square", "sawtooth")
TREND_KINDS: tuple[str, ...] = ("linear_drift", "exp_drift", "log_drift", "constant")


def white_noise(L: int, rng: np.random.Generator, std: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, std, size=L)


def red_noise(L: int, rng: np.random.Generator, std: float = 1.0) -> np.ndarray:
    """Cumulative-sum random walk, rescaled so terminal std ≈ ``std``."""
    steps = rng.normal(0.0, 1.0, size=L)
    walk = np.cumsum(steps)
    # Random walks have variance growing with t; normalize by sqrt(L) so the
    # series sits in a comparable scale to the other primitives.
    if L > 0:
        walk = walk / np.sqrt(L)
    return walk * std


def ar1(L: int, rng: np.random.Generator, phi: float = 0.7, std: float = 1.0) -> np.ndarray:
    """First-order autoregressive process with coefficient `phi`."""
    if not -1.0 < phi < 1.0:
        raise ValueError(f"AR(1) requires |phi| < 1, got {phi}")
    # Stationary innovation variance to keep marginal std ≈ std.
    sigma_eps = std * np.sqrt(1.0 - phi**2)
    eps = rng.normal(0.0, sigma_eps, size=L)
    out = np.empty(L, dtype=np.float64)
    out[0] = eps[0] / np.sqrt(1.0 - phi**2) if L > 0 else 0.0
    for i in range(1, L):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def sine(
    L: int,
    rng: np.random.Generator,
    period_range: tuple[float, float] = (16.0, 256.0),
    amp_range: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    period = rng.uniform(*period_range)
    amp = rng.uniform(*amp_range)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(L, dtype=np.float64)
    return amp * np.sin(2.0 * np.pi * t / period + phase)


def square(
    L: int,
    rng: np.random.Generator,
    period_range: tuple[float, float] = (16.0, 256.0),
    amp_range: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    period = rng.uniform(*period_range)
    amp = rng.uniform(*amp_range)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    t = np.arange(L, dtype=np.float64)
    return amp * np.sign(np.sin(2.0 * np.pi * t / period + phase))


def sawtooth(
    L: int,
    rng: np.random.Generator,
    period_range: tuple[float, float] = (16.0, 256.0),
    amp_range: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    period = rng.uniform(*period_range)
    amp = rng.uniform(*amp_range)
    phase = rng.uniform(0.0, period)
    t = np.arange(L, dtype=np.float64)
    # 2 * (frac - 0.5) maps [0,1) -> [-1,1).
    frac = ((t + phase) / period) % 1.0
    return amp * (2.0 * frac - 1.0)


def linear_drift(
    L: int,
    rng: np.random.Generator,
    slope_range: tuple[float, float] = (-1e-3, 1e-3),
) -> np.ndarray:
    slope = rng.uniform(*slope_range)
    t = np.arange(L, dtype=np.float64)
    return slope * t


def exp_drift(
    L: int,
    rng: np.random.Generator,
    rate_range: tuple[float, float] = (-1e-3, 1e-3),
) -> np.ndarray:
    rate = rng.uniform(*rate_range)
    t = np.arange(L, dtype=np.float64)
    # Center on 0 so this adds drift without an offset spike at t=0.
    return np.expm1(rate * t)


def log_drift(
    L: int,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.1, 2.0),
) -> np.ndarray:
    scale = rng.uniform(*scale_range)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    t = np.arange(L, dtype=np.float64)
    return sign * scale * np.log1p(t)


def constant(
    L: int,
    rng: np.random.Generator,
    value_range: tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    value = rng.uniform(*value_range)
    return np.full(L, value, dtype=np.float64)


_PRIMITIVE_REGISTRY = {
    "white_noise": white_noise,
    "red_noise": red_noise,
    "ar1": ar1,
    "sine": sine,
    "square": square,
    "sawtooth": sawtooth,
    "linear_drift": linear_drift,
    "exp_drift": exp_drift,
    "log_drift": log_drift,
    "constant": constant,
}


def draw_kind(kind: str, L: int, rng: np.random.Generator) -> np.ndarray:
    """Dispatch a primitive by registered name."""
    try:
        fn = _PRIMITIVE_REGISTRY[kind]
    except KeyError as e:
        raise KeyError(f"Unknown primitive {kind!r}; expected one of {sorted(_PRIMITIVE_REGISTRY)}") from e
    return fn(L, rng)
