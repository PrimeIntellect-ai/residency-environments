"""Classify-with-evidence (Tier II, evidence-grounded): identify the
anomaly's kind AND its window in a single JSON object.

Gold format: ``classify_with_evidence:{"kind": <label>, "start": int,
"end": int, "_L": int}``. The composite scorer
``kind_x_iou = kind_correct × iou`` is what enforces the grounding
requirement — a right label with the wrong window scores ``iou``, a
wrong label scores 0 regardless.

`kind` labels use the ARFBench Appendix C.2 wording.
"""

from __future__ import annotations

import json

import numpy as np

from ..anomalies import ANOMALY_KINDS, make_injection
from ..difficulty import Difficulty
from ..scene import channel_names
from .base import QuestionTemplate, SamplePlan

_CATEGORY = "classify_with_evidence"

# ARFBench Appendix C.2 wording for the anomaly kinds.
_KIND_TO_LABEL: dict[str, str] = {
    "level_shift": "Level Shift",
    "transient_spike": "Transient Spike",
    "change_in_seasonality": "Change in Seasonality",
    "change_in_variance": "Change in Variance",
    "change_in_trend": "Change in Trend",
}

# Comma-joined ARFBench labels, deterministic order — appears in the
# question text so the model knows the legal `kind` vocabulary.
_KIND_LABELS_LIST = ", ".join(f'"{label}"' for label in _KIND_TO_LABEL.values())

_QUESTION_TEMPLATE = (
    "Identify the anomaly in the time series and its window. State your "
    "final answer as a JSON object:\n"
    '{"kind": <label>, "start": <int>, "end": <int>}\n'
    f"`kind` must be one of: {_KIND_LABELS_LIST}. `start` is inclusive, "
    "`end` is exclusive. "
    "Each kind is defined as follows: "
    "a `Level Shift` is a sustained step in mean — values jump and remain shifted across the window; "
    "a `Transient Spike` is a brief impulse — one or a few samples diverge sharply, then return to baseline; "
    "a `Change in Variance` is a sustained increase (or decrease) in noise amplitude over a window with the mean unchanged; "
    "a `Change in Seasonality` is a sustained change in the periodic component's amplitude, frequency, or phase across a window; "
    "a `Change in Trend` is a slope change starting at the window — values drift up or down and remain on the new path afterwards. "
    "(Here L = __LENGTH__.)"
)


class ClassifyWithEvidence(QuestionTemplate):
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
            raise ValueError(f"ClassifyWithEvidence needs >=1 channel, got {n_channels}")
        difficulty = difficulty or Difficulty()

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
        payload = {
            "kind": _KIND_TO_LABEL[kind],
            "start": int(spec.t_start),
            "end": int(spec.t_end),
            "_L": int(length),
        }
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


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a classify_with_evidence gold: {gold!r}")
    return json.loads(gold[len(prefix) :])
