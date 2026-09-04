"""Tunable generation-difficulty knobs — the curriculum control surface.

A `Difficulty` overrides the per-sample ranges the generator otherwise draws from
fixed defaults. `Difficulty()` reproduces the benchmark distribution *exactly* (same
rng draw order → identical samples), so threading it in is a no-op until a controller
narrows/shifts the ranges. Carried per sample on `SampleSpec.difficulty`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .anomalies import MAX_WIDTH, MIN_WIDTH


@dataclass(frozen=True)
class Difficulty:
    magnitude_range: tuple[float, float] = (2.0, 6.0)
    """Anomaly magnitude in channel-std units. Smaller = subtler = harder."""
    width_range: tuple[int, int] = (MIN_WIDTH, MAX_WIDTH)
    """Anomaly width in samples. Narrower = harder to localize precisely."""
    noise_weights: Mapping[str, float] | None = None
    """Base-noise distribution bias over {white_noise, red_noise, ar1}; None = equal.
    Biasing toward red_noise / ar1 makes the baseline harder to separate from anomalies."""
    magnitude_log_range: tuple[float, float] = (1.0, 100.0)
    """measure_magnitude's answer magnitude, sampled log-uniformly. Narrower = harder."""
    max_k: int = 3
    """localize_all_channels anomaly count K ~ U{min_k..max_k}. More channels in play = harder."""
    min_k: int = 0
    """localize_all_channels anomaly count floor. 0 leaves ~25% of rows with empty gold;
    1 drops them, removing the always-`[]` attractor that collapses a cold-start policy."""
    lag_max_fraction: float = 0.25
    """lead_lag: max coupling lag = int(L * fraction). Larger spread = harder to align."""
    p_present: float = 0.8
    """localize: probability the row carries an injected anomaly. The remainder is the
    hard-negative absent case; 1.0 removes the always-`absent` attractor for RL warm-up."""
