"""AnomalyXL-precise taskset (Verifiers v1).

The data + scoring half of the environment. Tasks are synthetic multivariate
time-series with injected anomalies, generated on the fly: each one is
deterministic given ``(run_seed, index)``, and the series is regenerated in
``setup`` from the task's coordinates, so the long series never rides the wire.

The environment is decoupled from the harness: any v1 harness that executes
code (bash, codex, ...) can play it. The taskset provides tasks and scoring;
the harness provides the agent loop.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

import verifiers.v1 as vf

from ._anomalyxl import materialize, plan_from_dict
from .generator import GeneratorConfig, generate_sample
from .scoring import score_answer, score_metrics

__all__ = ["AnomalyXLConfig", "AnomalyXLData", "AnomalyXLTask", "AnomalyXLTaskset"]

VALUE_DECIMALS = 3
"""Decimals kept on channel values written to ``context.json``. Cuts the 131k x 16
worst case from ~41 MB to ~15 MB; the shift is ~1e-5 against a 10% scoring band."""


_VALUES_NOTE = (
    "Paired questions prefix channels with `A_` / `B_`. "
    "Multi-channel questions present 16 channels named `ch_00..ch_15`. "
)


def _round_series(series: dict[str, list[float]]) -> dict[str, list[float]]:
    """Snap channel values to ``VALUE_DECIMALS`` on the way to ``context.json``."""
    return {name: [round(v, VALUE_DECIMALS) for v in values] for name, values in series.items()}


def _build_note() -> str:
    return (
        "The time series is in `context.json` in your working directory, shaped as "
        '{"series": {channel: [values], ...}}. '
        + _VALUES_NOTE
        + "State your final answer as a single JSON object matching the question's "
        'schema (e.g. {"start": 12, "end": 87}).'
    )


class AnomalyXLData(vf.TaskData):
    """Wire data for one AnomalyXL-precise task.

    Carries the sample's plan so ``setup`` can build the series from it. Every
    field is required: a served task is rebuilt from this payload alone, so a
    field that went missing on the wire must fail validation rather than default
    into a different, plausible-looking series.
    """

    answer: str
    """AnomalyXL-encoded gold: ``f"{category}:{json}"``."""

    category: str
    """Question category (localize, classify_with_evidence, ...)."""

    plan: dict[str, Any]
    """Serialized ``SamplePlan`` — the anomalies, and the seeds that rebuild the
    series around them."""


class AnomalyXLTask(vf.Task[AnomalyXLData]):
    """One AnomalyXL-precise problem with scoring.

    ``setup`` builds the series from the task's plan and writes it as
    ``context.json`` in the runtime workspace. ``@reward`` scores the model's
    final reply; ``@metric`` reports per-category metrics.
    """

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        series = _round_series(materialize(plan_from_dict(self.data.plan)).series)
        await runtime.write("context.json", json.dumps({"series": series}).encode())

    @vf.reward(weight=1.0)
    async def correct_answer(self, trace: vf.Trace) -> float:
        """Primary reward: the category's primary metric on a 0-1 scale."""
        return score_answer(trace.last_reply, self.data.answer)

    @vf.metric
    async def extra_metrics(self, trace: vf.Trace) -> dict[str, float]:
        """This task's own metrics (IoU / F1 / ...), keyed ``<category>/<metric>`` so
        each one averages over the rows of its category alone."""
        return score_metrics(trace.last_reply, self.data.answer)


# arm64: nzumarraga/anomalyxl-precise.arm64@sha256:10a1ff49b94f52675bc64ad41d510b34f96c5026f8255bf9d94dc2a6a308742a
SANDBOX_IMAGE = (
    "nzumarraga/anomalyxl-precise.x86@sha256:f0ce634781f5cea957ae30344511936a876d3fb54b09fcdc6b6353faf950683a"
)
"""Ships numpy, scipy and matplotlib so the rollout never installs anything, which
is what lets the task run with the network blocked."""


class AnomalyXLConfig(vf.TasksetConfig):
    """Configuration for the AnomalyXL-precise taskset.

    Difficulty knobs live on ``generator.*``; ``generator.run_seed`` plus the task
    index defines the benchmark.
    """

    generator: GeneratorConfig = GeneratorConfig()
    """On-the-fly generator configuration."""


class AnomalyXLTaskset(vf.Taskset[AnomalyXLTask, AnomalyXLConfig]):
    """AnomalyXL-precise taskset. Tasks are deterministic given ``(run_seed, index)``."""

    def load(self) -> Iterator[AnomalyXLTask]:
        note = _build_note()
        cfg = self.config.generator

        for i in range(cfg.num_tasks):
            plan = generate_sample(cfg, i)
            data = AnomalyXLData(
                idx=i,
                prompt=f"{note}\n\n{plan.question}",
                answer=plan.answer,
                image=SANDBOX_IMAGE,
                network_allow=[],
                category=plan.category,
                plan=dataclasses.asdict(plan),
            )
            yield AnomalyXLTask(data, self.config.task)
