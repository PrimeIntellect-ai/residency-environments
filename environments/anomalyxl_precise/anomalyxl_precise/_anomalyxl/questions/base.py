"""Shared types for question templates.

A sample is produced in two steps. `plan` decides what the anomaly is and
derives the question and gold from that decision alone; `materialize` turns a
plan into the channel arrays. Only `materialize` is expensive, so a caller that
needs the gold — task loading — never builds a series.

`SamplePlan` is serializable: it is what rides on the task, and it is enough to
rebuild the exact same `Row`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..anomalies import CouplingSpec, InjectionSpec, inject
from ..scene import build_scene

__all__ = ["QuestionTemplate", "Row", "SamplePlan", "as_series"]


def as_series(channels: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Channel arrays as the plain lists that reach ``context.json``."""
    return {name: arr.astype(np.float32).tolist() for name, arr in channels.items()}


def plan_from_dict(data: dict) -> "SamplePlan":
    """Rebuild a plan from its ``dataclasses.asdict`` form."""
    coupling = data.get("coupling")
    split = data.get("paired_split")
    return SamplePlan(
        **{
            **data,
            "injections": [InjectionSpec(**spec) for spec in data.get("injections", [])],
            "coupling": CouplingSpec(**coupling) if coupling else None,
            "paired_split": tuple(split) if split else None,
        }
    )


@dataclass(frozen=True)
class Row:
    series: dict[str, list[float]]
    question: str
    answer: str
    injections: list[InjectionSpec] = field(default_factory=list)
    coupling: CouplingSpec | None = None


@dataclass(frozen=True)
class SamplePlan:
    """What a sample contains, decided without building any array."""

    category: str
    length: int
    n_channels: int
    question: str
    answer: str
    scene_seed: int
    injection_seed: int
    injections: list[InjectionSpec] = field(default_factory=list)
    coupling: CouplingSpec | None = None
    paired_split: tuple[float, float] | None = None
    noise_weights: dict[str, float] | None = None
    """Base-noise bias for `build_scene`; the only difficulty knob `materialize` reads."""


class QuestionTemplate(ABC):
    category: str
    tier: int
    # Minimum channel count this template can be built at. For paired (Tier III)
    # templates this is the *total* channel budget (A + B).
    min_channels: int = 1

    @abstractmethod
    def plan(
        self,
        length: int,
        n_channels: int,
        rng: np.random.Generator,
        *,
        scene_seed: int,
        injection_seed: int,
        paired_split: tuple[float, float] | None = None,
        difficulty: "object | None" = None,
    ) -> SamplePlan:
        """Decide the sample and derive its question and gold. Builds no arrays.

        Templates own the decision so they can condition it on the gold (Localize's
        absent case forbids any injection; ClassifyWithEvidence picks the kind
        first). Tier III templates split `n_channels` into `(A, B)` per
        `paired_split`.
        """

    def materialize(self, plan: SamplePlan) -> Row:
        """Build one scene and apply the plan's injections to it.

        Enough for every single-scene category. Paired (Tier III) templates build
        two prefixed scenes and override this.
        """
        scene = build_scene(
            plan.n_channels,
            plan.length,
            np.random.default_rng(plan.scene_seed),
            base_noise_weights=plan.noise_weights,
        )
        rng = np.random.default_rng(plan.injection_seed)
        for spec in plan.injections:
            scene.channels[spec.channel] = inject(scene.channels[spec.channel], spec, rng)
        return Row(
            series=as_series(scene.channels),
            question=plan.question,
            answer=plan.answer,
            injections=plan.injections,
            coupling=plan.coupling,
        )
