"""AnomalyXL-precise — evidence-grounded long-context anomaly QA.

A Verifiers v1 taskset that generates synthetic multivariate time-series with
injected anomalies on the fly and scores the model's JSON answer against
evidence-grounded gold. Five precise (open-ended, JSON) question families:

- ``localize`` — detect + locate an anomaly window (IoU)
- ``classify_with_evidence`` — identify anomaly kind + window (kind x IoU)
- ``measure_magnitude`` — quantify peak deviation in sigma-units (relative error)
- ``localize_all_channels`` — find all anomalies across channels (set F1)
- ``lead_lag_with_magnitude`` — temporal relationship + lag (composite)

Tasks are deterministic given ``(run_seed, index)``: ``generator.run_seed`` plus
the task index defines the benchmark.

The environment is decoupled from the harness: any v1 harness that executes
code (bash, codex, ...) can play it.
"""

from .generator import GeneratorCell, GeneratorConfig
from .taskset import AnomalyXLConfig, AnomalyXLData, AnomalyXLTask, AnomalyXLTaskset

__all__ = [
    "AnomalyXLConfig",
    "AnomalyXLData",
    "AnomalyXLTask",
    "AnomalyXLTaskset",
    "GeneratorCell",
    "GeneratorConfig",
]
