"""On-the-fly task generator for AnomalyXL-precise.

Replaces the static parquet with a lazy, deterministic sample stream from the
vendored anomalyxl generator. Each index maps deterministically (via blake2b on
``run_seed|i|salt``) to a generator cell and a sample seed, so ``tasks[i]`` is
reproducible and ``run_seed`` rotates the whole stream for fresh data per run.

A single fixed difficulty θ is pinned across the grid; per-task knobs ride on
the task record so ``setup`` can regenerate the series deterministically.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator
from pydantic_config import BaseConfig

from ._anomalyxl import TEMPLATES, Difficulty, SamplePlan, plan_sample

__all__ = ["GeneratorCell", "GeneratorConfig", "DEFAULT_CELLS"]


class GeneratorCell(BaseConfig):
    """One (category × channel count × length list) cell in the generation grid."""

    category: Literal[
        "localize",
        "classify_with_evidence",
        "measure_magnitude",
        "localize_all_channels",
        "lead_lag_with_magnitude",
    ]
    channels: int = Field(ge=1)
    lengths: list[int] = Field(min_length=1)

    weight: float = Field(default=1.0, ge=0.0)
    """Relative share of tasks drawn from this cell. Equal by default, so a cell's
    share does not depend on how many lengths it happens to list. Raise it to
    over-sample a category (a curriculum knob); 0 disables the cell."""

    @field_validator("lengths")
    @classmethod
    def _lengths_positive(cls, lengths: list[int]) -> list[int]:
        if any(length < 1 for length in lengths):
            raise ValueError(f"lengths must be positive, got {lengths}")
        return lengths

    @model_validator(mode="after")
    def _channels_fit_category(self) -> "GeneratorCell":
        required = TEMPLATES[self.category].min_channels
        if self.channels < required:
            raise ValueError(f"{self.category} needs >={required} channels, got {self.channels}")
        return self


DEFAULT_CELLS: tuple[GeneratorCell, ...] = (
    GeneratorCell(category="localize", channels=1, lengths=[16384, 32768, 65536, 131072]),
    GeneratorCell(category="classify_with_evidence", channels=1, lengths=[16384, 32768, 65536, 131072]),
    GeneratorCell(category="measure_magnitude", channels=1, lengths=[16384, 32768, 65536, 131072]),
    GeneratorCell(category="localize_all_channels", channels=16, lengths=[32768, 131072]),
    GeneratorCell(category="lead_lag_with_magnitude", channels=2, lengths=[32768, 131072]),
)


class GeneratorConfig(BaseConfig):
    """Configuration for the on-the-fly generator.

    ``cells`` defines the (category, channels, lengths) grid. Each task draws a
    cell by ``weight`` and then a length uniformly within it, so a category's
    share is set by its weight rather than by how many lengths it lists. The
    difficulty knobs below apply across all cells.
    """

    num_tasks: int = Field(default=100_000, ge=1)
    """Size of the lazy index space the taskset draws from."""

    run_seed: int = 0
    """Rotates the whole deterministic stream — bump for fresh data per run."""

    cells: list[GeneratorCell] = Field(default_factory=lambda: list(DEFAULT_CELLS), min_length=1)
    """The generation grid. Override to sub-select categories, channel counts, or lengths."""

    # --- Anomaly difficulty knobs ---

    magnitude_range: tuple[float, float] = (2.0, 6.0)
    """Anomaly magnitude in channel-std units. Smaller = subtler = harder."""

    width_range: tuple[int, int] = (8, 256)
    """Anomaly width in samples. Narrower = harder to localize precisely."""

    width_frac_range: tuple[float, float] | None = None
    """RELATIVE anomaly width: when set, each task's absolute ``width_range`` is
    derived as this fraction of its series length (floored at 8, capped at L/8).
    Decouples relative width from length. Keep ≲ 0.05 to avoid raising the
    guessing baseline. None = absolute ``width_range``."""

    magnitude_log_range: tuple[float, float] = (1.0, 100.0)
    """measure_magnitude's answer magnitude, sampled log-uniformly."""

    noise_weights: dict[str, float] | None = None
    """Base-noise distribution bias over {white_noise, red_noise, ar1}; None = equal.
    Biasing toward red_noise/ar1 makes the baseline harder to separate from anomalies."""

    max_k: int = Field(default=3, ge=0)
    """localize_all_channels anomaly count K ~ U{min_k..max_k}."""

    min_k: int = Field(default=0, ge=0)
    """localize_all_channels anomaly count floor. Set to 1 for RL warm-up to drop the
    empty-gold rows; stage back to 0 for calibration."""

    lag_max_fraction: float = Field(default=0.25, ge=0.0, lt=1.0)
    """lead_lag: max coupling lag = int(L * fraction). Larger spread = harder."""

    p_present: float = Field(default=0.8, ge=0.0, le=1.0)
    """Probability the localize category injects an anomaly (vs. the hard-negative
    absent case). Set to 1.0 for RL warm-up; stage back to 0.8 for calibration."""

    @field_validator("magnitude_range", "width_range", "width_frac_range", "magnitude_log_range")
    @classmethod
    def _ordered_and_positive(cls, value: tuple[float, float] | None, info) -> tuple[float, float] | None:
        if value is None:
            return value
        lo, hi = value
        if lo <= 0:
            raise ValueError(f"{info.field_name} bounds must be positive, got {value}")
        if lo > hi:
            raise ValueError(f"{info.field_name} must be ordered (lo <= hi), got {value}")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> "GeneratorConfig":
        if self.min_k > self.max_k:
            raise ValueError(f"min_k ({self.min_k}) must not exceed max_k ({self.max_k})")
        if not any(cell.weight > 0 for cell in self.cells):
            raise ValueError("at least one cell must have a positive weight")
        # place_k_anomalies puts each anomaly on its own channel.
        for cell in self.cells:
            if cell.category == "localize_all_channels" and self.max_k > cell.channels:
                raise ValueError(f"max_k ({self.max_k}) exceeds the {cell.channels} channels of a {cell.category} cell")
        return self


def _theta(cfg: GeneratorConfig) -> Difficulty:
    return Difficulty(
        magnitude_range=cfg.magnitude_range,
        width_range=cfg.width_range,
        magnitude_log_range=cfg.magnitude_log_range,
        noise_weights=cfg.noise_weights,
        max_k=cfg.max_k,
        min_k=cfg.min_k,
        lag_max_fraction=cfg.lag_max_fraction,
        p_present=cfg.p_present,
    )


def _seed(run_seed: int, i: int, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{run_seed}|{i}|{salt}".encode(), digest_size=8).digest(), "big")


def resolve_theta(cfg: GeneratorConfig, category: str, length: int) -> Difficulty:
    """Per-task θ: the fixed base difficulty, with relative width resolved against
    this task's length. ``width_frac_range`` wins over any absolute ``width_range``."""
    theta = _theta(cfg)
    frac = cfg.width_frac_range
    if frac is not None:
        lo = max(8, min(int(frac[0] * length), length // 8))
        hi = max(lo + 1, min(int(frac[1] * length), length // 8))
        theta = dataclasses.replace(theta, width_range=(lo, hi))
    return theta


def _pick_cell(cells: list[GeneratorCell], rng: np.random.Generator) -> GeneratorCell:
    """Draw a cell by weight. Sampling the cell before its length keeps a cell's
    share independent of how many lengths it lists."""
    weights = np.array([c.weight for c in cells], dtype=float)
    if (weights < 0).any():
        raise ValueError(f"cell weights must be non-negative, got {list(weights)}")
    total = weights.sum()
    if total <= 0:
        raise ValueError("at least one cell must have a positive weight")
    return cells[int(rng.choice(len(cells), p=weights / total))]


def generate_sample(cfg: GeneratorConfig, index: int) -> SamplePlan:
    """Deterministically resolve one task's plan from its index.

    Builds no arrays: the plan carries the question, the gold, and the seeds
    ``materialize`` needs to reproduce the series exactly.
    """
    rng = np.random.default_rng(_seed(cfg.run_seed, index, "pick"))
    cell = _pick_cell(cfg.cells, rng)
    length = cell.lengths[int(rng.integers(0, len(cell.lengths)))]
    category = cell.category
    return plan_sample(
        category,
        length,
        cell.channels,
        plan_seed=_seed(cfg.run_seed, index, "plan"),
        scene_seed=_seed(cfg.run_seed, index, "scene"),
        injection_seed=_seed(cfg.run_seed, index, "injection"),
        paired_split=(0.5, 0.5) if category == "lead_lag_with_magnitude" else None,
        difficulty=resolve_theta(cfg, category, length),
    )
