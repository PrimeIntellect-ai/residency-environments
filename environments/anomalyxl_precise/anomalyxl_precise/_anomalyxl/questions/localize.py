"""Localize (Tier II, evidence-grounded): does the series contain an
anomaly, and if so, what is its window?

The model commits to *both* presence and grounding in a single JSON
object: `{"present": bool, "start": int, "end": int}`. Gold is sampled
present with probability `Difficulty.p_present` (0.8 by default) — the
absent case is essential so a precision-style false-positive penalty
exists, but a uniform 50/50 split over-weights the trivial "say no"
baseline and dilutes the localization signal. 20% absent keeps the
precision pressure while leaving 80% of rows scoring grounded IoU.

Gold envelope is the AnomalyXL convention:
``f"localize:{json.dumps({...})}"`` where the payload always carries
`_L` so the env scorer can compute MAE fractions without re-reading
the parquet `length` column.
"""

from __future__ import annotations

import json

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection
from ..difficulty import Difficulty
from ..scene import channel_names
from .base import QuestionTemplate, SamplePlan

_CATEGORY = "localize"

_QUESTION_TEMPLATE = (
    "Does the time series contain an anomaly, and if so where? State your "
    "final answer as a JSON object:\n"
    '{"present": true|false, "start": <int>, "end": <int>}\n'
    'If "present" is false, "start" and "end" are ignored. If true, '
    "report the inclusive start and exclusive end integer sample indices "
    "of the anomaly. (Here L = __LENGTH__.)"
)


class Localize(QuestionTemplate):
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
            raise ValueError(f"Localize needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

        present = bool(rng.uniform() < difficulty.p_present)
        injections = []
        if present:
            target = channel_names(n_channels)[int(rng.integers(0, n_channels))]
            kind = ANOMALY_KINDS[int(rng.integers(0, len(ANOMALY_KINDS)))]
            spec = make_injection(
                length,
                rng,
                kind=kind,
                channel=target,
                min_width=difficulty.width_range[0],
                max_width=difficulty.width_range[1],
                magnitude_range=difficulty.magnitude_range,
            )
            injections.append(spec)
            payload = {
                "present": True,
                "start": int(spec.t_start),
                "end": int(spec.t_end),
                "_L": int(length),
            }
        else:
            payload = {"present": False, "_L": int(length)}

        return SamplePlan(
            category=_CATEGORY,
            length=length,
            n_channels=n_channels,
            question=_QUESTION_TEMPLATE.replace("__LENGTH__", str(length)),
            answer=f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}",
            scene_seed=scene_seed,
            injection_seed=injection_seed,
            injections=injections,
            noise_weights=difficulty.noise_weights,
        )


def _parse_gold_payload(gold: str) -> dict:
    """Recover the JSON payload from a ``localize:<json>`` gold string."""
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a localize gold: {gold!r}")
    return json.loads(gold[len(prefix) :])
