"""Lead-lag-with-magnitude (Tier III paired, evidence-grounded):
identify the temporal relationship between anomalies in two
single-channel sub-scenes A and B, AND quantify the lag.

Gold envelope:
``lead_lag_with_magnitude:{"direction": <"lead"|"lag"|"independent">,
                          "lag_samples": <int>, "_L": <int>}``.

`lag_samples` is always non-negative — it's the *magnitude* of the
offset between A's and B's anomaly start indices. The direction tag
disambiguates sign:
- ``"lead"``: A's anomaly precedes B's by ``lag_samples`` (i.e. A is
  the leading indicator).
- ``"lag"``: A's anomaly follows B's by ``lag_samples`` (A is the
  lagging indicator).
- ``"independent"``: both sides contain anomalies in disjoint windows;
  ``lag_samples`` is ignored.

Direction is sampled 1/3 / 1/3 / 1/3 per plan §3.5; for lead/lag the
magnitude is uniform in ``[1, L/4]``. C is fixed at 2 (one channel per
side) — keeps the prompt narrow and the gold unambiguous.

Reuses ``_paired.build_paired_scenes`` for the A/B scaffold and
``_paired.inject_for_coupling``'s ``independent`` branch for the
no-relationship case; the coupled cases inline the per-side injection
so we can pin ``lag_samples`` exactly without going through
``make_coupling``'s own lag sampler.
"""

from __future__ import annotations

import json

import numpy as np

from ..anomalies import (
    MAX_WIDTH,
    CouplingSpec,
    inject,
)
from ..difficulty import Difficulty
from ..scene import build_scene, channel_names
from ._paired import plan_for_coupling, split_channels
from .base import QuestionTemplate, Row, SamplePlan, as_series

_CATEGORY = "lead_lag_with_magnitude"

_DIRECTIONS: tuple[str, ...] = ("lead", "lag", "independent")

_QUESTION_TEMPLATE = (
    "Two single-channel sub-series are shown, with channels prefixed "
    "`A_` and `B_`. Both contain an anomaly (or independent anomalies). "
    "Is A a leading or lagging indicator of B, and by how many samples? "
    "State your final answer as a JSON object:\n"
    '{"direction": "lead"|"lag"|"independent", "lag_samples": <int>}\n'
    "`lag_samples` is the non-negative magnitude (in sample indices) of "
    "the offset between A's and B's anomaly start; it is ignored in the "
    "independent case. "
    "Two anomalies are considered independent when they occupy unrelated "
    "windows with no causal alignment; by construction, coupled lead/lag "
    "offsets never exceed L/4 samples. (Here L = __LENGTH__.)"
)


class LeadLagWithMagnitude(QuestionTemplate):
    category = _CATEGORY
    tier = 3
    min_channels = 2

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
        if n_channels < self.min_channels:
            raise ValueError(f"LeadLagWithMagnitude needs >={self.min_channels} channels, got {n_channels}")
        difficulty = difficulty or Difficulty()

        a, b = split_channels(n_channels, paired_split)
        names_a = channel_names(a, "A_")
        names_b = channel_names(b, "B_")
        direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]

        if direction == "independent":
            coupling = CouplingSpec(kind="independent", lag=0)
            injections, coupling = plan_for_coupling(names_a, names_b, length, coupling, rng, difficulty=difficulty)
            lag_samples = 0
        else:
            # Pin lag magnitude in [1, L/4]; sign comes from direction.
            max_lag = min(max(1, int(length * difficulty.lag_max_fraction)), length - MAX_WIDTH)
            if max_lag < 1:
                raise ValueError(f"L={length} is too short for any lag with MAX_WIDTH={MAX_WIDTH}")
            lag_samples = int(rng.integers(1, max_lag + 1))
            signed_lag = lag_samples if direction == "lead" else -lag_samples
            coupling = CouplingSpec(
                kind="coupled_lag_pos" if direction == "lead" else "coupled_lag_neg",
                lag=signed_lag,
            )
            injections, coupling = plan_for_coupling(names_a, names_b, length, coupling, rng, difficulty=difficulty)
            if coupling.kind == "coupled_simul":
                raise RuntimeError(
                    f"plan_for_coupling degraded to simultaneous at L={length}, "
                    f"lag={signed_lag}. Tighten max_lag bound."
                )
            lag_samples = abs(coupling.lag)

        payload = {"direction": direction, "lag_samples": int(lag_samples), "_L": int(length)}
        return SamplePlan(
            category=_CATEGORY,
            length=length,
            n_channels=n_channels,
            question=_QUESTION_TEMPLATE.replace("__LENGTH__", str(length)),
            answer=f"{_CATEGORY}:{json.dumps(payload, sort_keys=True)}",
            scene_seed=scene_seed,
            injection_seed=injection_seed,
            injections=injections,
            coupling=coupling,
            paired_split=paired_split,
            noise_weights=difficulty.noise_weights,
        )

    def materialize(self, plan: SamplePlan) -> Row:
        a, b = split_channels(plan.n_channels, plan.paired_split)
        scene_rng = np.random.default_rng(plan.scene_seed)
        scene_a = build_scene(a, plan.length, scene_rng, name_prefix="A_", base_noise_weights=plan.noise_weights)
        scene_b = build_scene(b, plan.length, scene_rng, name_prefix="B_", base_noise_weights=plan.noise_weights)

        rng = np.random.default_rng(plan.injection_seed)
        for spec in plan.injections:
            scene = scene_a if spec.channel in scene_a.channels else scene_b
            scene.channels[spec.channel] = inject(scene.channels[spec.channel], spec, rng)

        return Row(
            series={**as_series(scene_a.channels), **as_series(scene_b.channels)},
            question=plan.question,
            answer=plan.answer,
            injections=plan.injections,
            coupling=plan.coupling,
        )


def _parse_gold_payload(gold: str) -> dict:
    prefix = f"{_CATEGORY}:"
    if not gold.startswith(prefix):
        raise ValueError(f"Not a lead_lag_with_magnitude gold: {gold!r}")
    return json.loads(gold[len(prefix) :])
