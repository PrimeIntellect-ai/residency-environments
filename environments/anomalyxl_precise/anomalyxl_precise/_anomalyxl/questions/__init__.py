"""Question templates registry.

Each `QuestionTemplate` subclass implements `build(length, n_channels, rng,
*, paired_split=None)` and is registered in `TEMPLATES` so the pipeline can
dispatch by `category`.

Only the five *precise* families are registered — the open-ended AnomalyXL
tasks with strict JSON answers and continuous scores. The released dataset's
`MCQ` subset is built from a different set of templates, which this
precise-only environment cannot score.
"""

from __future__ import annotations

from .base import QuestionTemplate, Row, SamplePlan, as_series, plan_from_dict
from .classify_with_evidence import ClassifyWithEvidence
from .lead_lag_with_magnitude import LeadLagWithMagnitude
from .localize import Localize
from .localize_all_channels import LocalizeAllChannels
from .measure_magnitude import MeasureMagnitude

TEMPLATES: dict[str, type[QuestionTemplate]] = {
    cls.category: cls
    for cls in (
        Localize,
        ClassifyWithEvidence,
        MeasureMagnitude,
        LocalizeAllChannels,
        LeadLagWithMagnitude,
    )
}

__all__ = ["QuestionTemplate", "Row", "SamplePlan", "TEMPLATES", "as_series", "plan_from_dict"]
