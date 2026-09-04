"""Measure-magnitude (Tier II, evidence-grounded): report the anomaly's
peak deviation in σ-units of the baseline (pre-injection) channel std.

Implementation contract:
- Always inject a single ``level_shift`` so peak deviation = exactly
  ``magnitude × pre_std`` analytically (no envelope, no decay, no
  stochastic peak).
- ``magnitude`` is sampled **log-uniformly** in ``[1.0, 100.0]`` —
  matches the ``acc_at_rel_*`` band spacing and stresses both
  near-floor (~1σ) and far (~100σ) ranges.
- The gold ``magnitude_sigma`` is the sampled magnitude. A level shift
  adds ``magnitude × channel_std`` over the window and the gold is the
  peak deviation in units of that same std, so the two cancel: measuring
  ``peak / pre_std`` off the built arrays agrees to within 4e-7 relative,
  against scoring bands of 10%. Deriving it in ``plan`` is what lets the
  gold exist before any array does.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..anomalies import InjectionSpec
from ..difficulty import Difficulty
from ..scene import channel_names
from .base import QuestionTemplate, SamplePlan

_CATEGORY = "measure_magnitude"

_MAGNITUDE_LO: float = 1.0
_MAGNITUDE_HI: float = 100.0

_QUESTION_TEMPLATE = (
    "An anomaly is present in the time series. Report its peak deviation "
    "in units of the channel's baseline (pre-anomaly) standard deviation "
    "(σ). State your final answer as a JSON object:\n"
    '{"magnitude_sigma": <float>}\n'
    "Use a positive value (the deviation magnitude, not its sign). "
    "(Here L = __LENGTH__.)"
)


class MeasureMagnitude(QuestionTemplate):
    category = _CATEGORY
    tier = 2
    min_channels = 1

    def plan(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        scene_seed: int,
        injection_seed: int,
        paired_split: tuple[float, float] | None = None,
        difficulty: "Difficulty | None" = None,
    ) -> SamplePlan:
        del paired_split
        if n_channels < 1:
            raise ValueError(f"MeasureMagnitude needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

        target = channel_names(n_channels)[int(rng.integers(0, n_channels))]
        magnitude = _sample_log_uniform(rng, difficulty.magnitude_log_range[0], difficulty.magnitude_log_range[1])
        max_w = min(difficulty.width_range[1], length)
        width = int(rng.integers(difficulty.width_range[0], max(difficulty.width_range[0] + 1, max_w + 1)))
        t_start = int(rng.integers(0, max(0, length - width) + 1))
        spec = InjectionSpec(
            channel=target,
            t_start=t_start,
            t_end=t_start + width,
            kind="level_shift",
            magnitude=float(magnitude),
        )
        # A level shift adds `magnitude * channel_std` over the window, and the gold is
        # the peak deviation in units of that same std — so the gold is the magnitude.
        payload = {"magnitude_sigma": round(float(magnitude), 6), "_L": int(length)}
        return SamplePlan(
            category=_CATEGORY,
            length=length,
            n_channels=n_channels,
            question=_QUESTION_TEMPLATE.replace("__LENGTH__", str(length)),
            answer=f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}",
            scene_seed=scene_seed,
            injection_seed=injection_seed,
            injections=[spec],
            noise_weights=difficulty.noise_weights,
        )


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw `x ~ exp(U(log lo, log hi))`. lo and hi must be positive."""
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return math.exp(float(rng.uniform(log_lo, log_hi)))


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a measure_magnitude gold: {gold!r}")
    return json.loads(gold[len(prefix) :])
