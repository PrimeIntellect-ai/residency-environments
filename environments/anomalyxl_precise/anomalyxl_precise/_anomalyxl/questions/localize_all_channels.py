"""Localize-all-channels (Tier III, multi-channel evidence-grounded):
report every anomaly across C channels as `{channel, start, end}` events.

K (anomalies per row) is sampled uniformly in
`{Difficulty.min_k .. Difficulty.max_k}`, `{0, 1, 2, 3}` by default —
K=0 is essential so a precision-style false-positive penalty is
meaningful. Each anomaly lives on a *distinct* channel (per the
`_multi_injection.place_k_anomalies` contract).

The template is registered with `min_channels=16`; ablations on smaller
channel subsets happen post-hoc at eval time by filtering events.

Gold payload always contains an `anomalies` list (possibly empty) plus
`_L`. Events are sorted by `(channel, start)` for stable equality in
tests and reproducibility.
"""

from __future__ import annotations

import json

import numpy as np

from ..difficulty import Difficulty
from ..scene import channel_names
from ._multi_injection import plan_k_anomalies
from .base import QuestionTemplate, SamplePlan

_CATEGORY = "localize_all_channels"

_MAX_K: int = 3

_QUESTION_TEMPLATE = (
    "The time series has multiple channels. Find every anomaly across "
    "all channels. State your final answer as a JSON object:\n"
    '{"anomalies": [{"channel": "<name>", "start": <int>, "end": <int>}, ...]}\n'
    'Report an empty list `"anomalies": []` if no anomalies are present. '
    "`channel` must match a channel name from the input verbatim. `start` is "
    "inclusive, `end` is exclusive. (Here L = __LENGTH__, C = __N_CHANNELS__.)"
)


class LocalizeAllChannels(QuestionTemplate):
    category = _CATEGORY
    tier = 3
    min_channels = 16

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
        if n_channels < self.min_channels:
            raise ValueError(f"LocalizeAllChannels needs >={self.min_channels} channels, got {n_channels}")
        difficulty = difficulty or Difficulty()

        k = int(rng.integers(difficulty.min_k, difficulty.max_k + 1))
        injections = plan_k_anomalies(channel_names(n_channels), length, k, rng, difficulty=difficulty)

        # Sort events by (channel, start) so gold is order-stable.
        events = sorted(
            ({"channel": s.channel, "start": int(s.t_start), "end": int(s.t_end)} for s in injections),
            key=lambda e: (e["channel"], e["start"]),
        )
        payload = {"anomalies": events, "_L": int(length)}
        question = _QUESTION_TEMPLATE.replace("__LENGTH__", str(length)).replace("__N_CHANNELS__", str(n_channels))
        return SamplePlan(
            category=_CATEGORY,
            length=length,
            n_channels=n_channels,
            question=question,
            answer=f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}",
            scene_seed=scene_seed,
            injection_seed=injection_seed,
            injections=injections,
            noise_weights=difficulty.noise_weights,
        )


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a localize_all_channels gold: {gold!r}")
    return json.loads(gold[len(prefix) :])
