"""Vendored anomalyxl generator — synthetic multivariate anomaly QA.

Produces synthetic time-series with injected anomalies and evidence-grounded
gold answers. Five precise (open-ended, JSON) question families:

- ``localize`` — detect + locate an anomaly window (IoU score)
- ``classify_with_evidence`` — identify anomaly kind + window (kind × IoU)
- ``measure_magnitude`` — quantify peak deviation in σ-units (relative error)
- ``localize_all_channels`` — find all anomalies across channels (set F1)
- ``lead_lag_with_magnitude`` — temporal relationship + lag (composite)
"""

from __future__ import annotations

from .anomalies import (
    ANOMALY_KINDS,
    COUPLING_KINDS,
    CouplingSpec,
    InjectionSpec,
    inject,
    make_coupling,
    make_injection,
)
from .difficulty import Difficulty
from .pipeline import Row, SamplePlan, materialize, plan_sample
from .questions import TEMPLATES, QuestionTemplate, plan_from_dict
from .scene import Scene, build_scene

__all__ = [
    "ANOMALY_KINDS",
    "COUPLING_KINDS",
    "CouplingSpec",
    "Difficulty",
    "InjectionSpec",
    "QuestionTemplate",
    "Row",
    "SamplePlan",
    "Scene",
    "TEMPLATES",
    "build_scene",
    "materialize",
    "plan_from_dict",
    "plan_sample",
    "inject",
    "make_coupling",
    "make_injection",
]
