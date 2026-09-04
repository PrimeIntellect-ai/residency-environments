"""Multivariate scene assembly: many independent channels built from
`primitives.py` building blocks.

Channels are anonymous (`ch_00..ch_NN` or prefixed `A_ch_NN` / `B_ch_NN`) so
the model can't piggy-back on channel-name priors. Each channel is the sum of
one base noise + optional periodic + optional trend, then z-normalized to
unit-ish std so anomalies expressed as multiples of `channel_std` stay
visible regardless of base mix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from . import primitives as P

__all__ = ["Scene", "build_scene", "channel_names", "DEFAULT_PERIODIC_P", "DEFAULT_TREND_P"]


DEFAULT_PERIODIC_P: float = 0.4
DEFAULT_TREND_P: float = 0.3


@dataclass(frozen=True)
class Scene:
    """A multivariate scene before any anomaly is injected.

    `channels` keeps insertion order. `base_components` records exactly which
    primitives went into each channel — useful for tests and downstream
    debugging, never read by the question templates themselves.
    """

    channels: dict[str, np.ndarray]
    base_components: dict[str, list[str]]

    @property
    def length(self) -> int:
        if not self.channels:
            return 0
        return next(iter(self.channels.values())).shape[0]

    @property
    def names(self) -> list[str]:
        return list(self.channels.keys())


def _channel_name(idx: int, prefix: str, total: int) -> str:
    """`ch_00`, `ch_01`, ... `ch_NN` — pad to the natural width of `total-1`."""
    width = max(2, len(str(max(0, total - 1))))
    return f"{prefix}ch_{idx:0{width}d}"


def channel_names(n_channels: int, prefix: str = "") -> list[str]:
    """The names `build_scene` will give these channels — derivable without building them."""
    return [_channel_name(i, prefix, n_channels) for i in range(n_channels)]


def _normalize_unit(x: np.ndarray) -> np.ndarray:
    """Center and rescale to unit std (no-op if std ≈ 0)."""
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-9:
        return x - mean
    return (x - mean) / std


def build_scene(
    n_channels: int,
    length: int,
    rng: np.random.Generator,
    *,
    periodic_p: float = DEFAULT_PERIODIC_P,
    trend_p: float = DEFAULT_TREND_P,
    base_noise_weights: Mapping[str, float] | None = None,
    name_prefix: str = "",
) -> Scene:
    """Build `n_channels` independent channels of `length` samples each.

    Each channel = (one base noise) + (periodic with prob `periodic_p`)
    + (trend with prob `trend_p`), then z-normalized so std ≈ 1.

    `base_noise_weights` lets a caller bias the base-noise distribution; by
    default the three kinds are equally likely. `name_prefix` is "" for a
    single scene, "A_" or "B_" for paired scenes (Tier III).
    """
    if n_channels < 0:
        raise ValueError(f"n_channels must be non-negative, got {n_channels}")
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")

    noise_kinds = list(P.BASE_NOISE_KINDS)
    if base_noise_weights is None:
        weights = np.full(len(noise_kinds), 1.0 / len(noise_kinds))
    else:
        raw = np.array([float(base_noise_weights.get(k, 0.0)) for k in noise_kinds])
        total = raw.sum()
        if total <= 0:
            raise ValueError("base_noise_weights must sum to > 0 over known kinds")
        weights = raw / total

    channels: dict[str, np.ndarray] = {}
    base_components: dict[str, list[str]] = {}

    for i in range(n_channels):
        name = _channel_name(i, name_prefix, n_channels)
        used: list[str] = []

        # Base noise: always exactly one.
        base_kind = noise_kinds[int(rng.choice(len(noise_kinds), p=weights))]
        series = P.draw_kind(base_kind, length, rng)
        used.append(base_kind)

        # Optional periodic component.
        if rng.uniform() < periodic_p:
            periodic_kind = P.PERIODIC_KINDS[int(rng.integers(0, len(P.PERIODIC_KINDS)))]
            series = series + P.draw_kind(periodic_kind, length, rng)
            used.append(periodic_kind)

        # Optional trend component.
        if rng.uniform() < trend_p:
            trend_kind = P.TREND_KINDS[int(rng.integers(0, len(P.TREND_KINDS)))]
            series = series + P.draw_kind(trend_kind, length, rng)
            used.append(trend_kind)

        channels[name] = _normalize_unit(series)
        base_components[name] = used

    return Scene(channels=channels, base_components=base_components)
