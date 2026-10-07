from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.metrics import (
    baseline_normalized_score,
    row_overlap_metrics,
    weighted_match_score,
)
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class PipelineEvaluator(BaseEvaluator):
    CAPABILITY_WEIGHTS = {
        "execution": 0.05,
        "top_level_shape": 0.10,
        "row_schema": 0.15,
        "normalization": 0.15,
        "filtering": 0.15,
        "aggregation": 0.25,
        "ordering": 0.05,
        "determinism": 0.10,
    }

    def capability_scores(
        self,
        expected: object,
        actual: object,
        *,
        deterministic_rerun: float,
    ) -> dict[str, float]:
        expected_rows = expected if isinstance(expected, list) else []
        top_level_shape = float(isinstance(actual, list))
        raw_rows = actual if isinstance(actual, list) else actual.get("rows", []) if isinstance(actual, dict) else []
        actual_rows = [row for row in raw_rows if isinstance(row, dict)]
        expected_key_set = {"team", "job_count", "total_hours"}
        row_schema = float(
            bool(actual_rows)
            and len(actual_rows) == len(raw_rows)
            and all(set(row) == expected_key_set for row in actual_rows)
        )
        legacy_schema = bool(actual_rows) and all(set(row) == {"team", "jobs", "hours"} for row in actual_rows)
        semantic_factor = 0.75 if legacy_schema or not top_level_shape else 1.0
        canonical_rows = [
            {
                "team": row.get("team"),
                "job_count": row.get("job_count", row.get("jobs")),
                "total_hours": row.get("total_hours", row.get("hours")),
            }
            for row in actual_rows
        ]
        expected_teams = [str(row.get("team")) for row in expected_rows if isinstance(row, dict)]
        actual_teams = [str(row.get("team")) for row in canonical_rows]
        normalization = semantic_factor * float(sorted(actual_teams) == sorted(expected_teams) and bool(expected_teams))
        expected_count = sum(int(row.get("job_count", 0)) for row in expected_rows if isinstance(row, dict))
        actual_count = sum(int(row.get("job_count") or 0) for row in canonical_rows)
        filtering = semantic_factor * float(actual_count == expected_count and expected_count > 0)
        expected_hours = {
            str(row.get("team")): row.get("total_hours") for row in expected_rows if isinstance(row, dict)
        }
        actual_hours = {str(row.get("team")): row.get("total_hours") for row in canonical_rows}
        aggregation = semantic_factor * float(actual_hours == expected_hours and bool(expected_hours))
        ordering = semantic_factor * float(actual_teams == expected_teams and bool(actual_teams))
        return {
            "execution": 1.0,
            "top_level_shape": top_level_shape,
            "row_schema": row_schema,
            "normalization": normalization,
            "filtering": filtering,
            "aggregation": aggregation,
            "ordering": ordering,
            "determinism": deterministic_rerun,
        }

    def weighted_capability_score(self, scores: dict[str, float]) -> float:
        return round(
            sum(self.CAPABILITY_WEIGHTS[name] * scores.get(name, 0.0) for name in self.CAPABILITY_WEIGHTS),
            6,
        )

    def required_json_artifact_scores(
        self,
        workspace_path: Path,
        hidden_root: Path,
        config: dict[str, object],
    ) -> tuple[dict[str, float], list[str]]:
        scores: dict[str, float] = {}
        failures: list[str] = []
        for raw_item in config.get("required_json_artifacts", []):
            item = dict(raw_item)
            capability = str(item.get("capability", "required_artifact"))
            try:
                actual = json.loads((workspace_path / str(item["path"])).read_text(encoding="utf-8"))
                expected = read_json(hidden_root / str(item["expected_path"]))
                score = float(actual == expected)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                score = 0.0
            scores[capability] = score
            if score == 0.0:
                failures.append(f"{capability}_failed")
        return scores, failures

    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        config = read_json(hidden_root / "evaluator_config.json")
        precomputed_path = hidden_root / "precomputed_execution.json"
        if not precomputed_path.is_file():
            raise ValueError("pipeline execution must be collected in the isolated runner runtime")
        run = dict(read_json(precomputed_path).get("visible", {}))
        completed = SimpleNamespace(
            returncode=int(run.get("returncode", 1)),
            stdout=str(run.get("stdout", "")),
            stderr=str(run.get("stderr", "")),
        )
        if completed.returncode != 0:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={
                    "execution": 0.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["execution_failed"],
                diagnostics={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                runtime_seconds=time.perf_counter() - started,
            )

        output_path = workspace_path / config["required_output_path"]
        if not output_path.exists():
            raw_score = 0.05
            score = (
                baseline_normalized_score(raw_score, float(config["initial_score"]))
                if config.get("initial_score") is not None
                else raw_score
            )
            return EvaluatorResult(
                success=False,
                score=score,
                subscores={
                    "execution": 1.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["output_missing"],
                diagnostics={"required_output_path": config["required_output_path"]},
                runtime_seconds=time.perf_counter() - started,
            )

        try:
            actual = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raw_score = 0.10
            score = (
                baseline_normalized_score(raw_score, float(config["initial_score"]))
                if config.get("initial_score") is not None
                else raw_score
            )
            return EvaluatorResult(
                success=False,
                score=score,
                subscores={
                    "execution": 1.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["invalid_json"],
                diagnostics={"error": str(exc)},
                runtime_seconds=time.perf_counter() - started,
            )

        expected = read_json(hidden_root / "expected_output.json")
        rerun = SimpleNamespace(returncode=int(run.get("rerun_returncode", 1)))
        deterministic_rerun = float(bool(run.get("deterministic", False)))
        artifact_scores, artifact_failures = self.required_json_artifact_scores(workspace_path, hidden_root, config)
        artifacts_valid = all(score == 1.0 for score in artifact_scores.values())
        success = actual == expected and deterministic_rerun == 1.0 and artifacts_valid
        metrics = row_overlap_metrics(expected, actual)
        score = weighted_match_score(output_exists=1.0, valid_structure=1.0, metrics=metrics)
        capability_scores: dict[str, float] = {}
        if config.get("capability_scoring"):
            capability_scores = self.capability_scores(expected, actual, deterministic_rerun=deterministic_rerun)
            capability_scores.update(artifact_scores)
            score = self.weighted_capability_score(capability_scores)
        if artifact_failures:
            score = min(score, float(config.get("required_artifact_failure_cap", 0.30)))
        raw_score = score
        if config.get("initial_score") is not None:
            score = baseline_normalized_score(raw_score, float(config["initial_score"]))
        if deterministic_rerun != 1.0:
            score = 0.0
        if success:
            score = 1.0
        diagnostics = {
            "required_output_path": config["required_output_path"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "rerun_returncode": rerun.returncode,
            "deterministic_rerun": bool(deterministic_rerun),
            "artifact_failures": artifact_failures,
            "raw_score": raw_score,
            "initial_score": config.get("initial_score"),
        }
        if not success:
            diagnostics["expected_preview"] = expected[:2] if isinstance(expected, list) else expected
            diagnostics["actual_preview"] = actual[:2] if isinstance(actual, list) else actual
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={
                "execution": 1.0,
                "valid_json": 1.0,
                "deterministic_rerun": deterministic_rerun,
                **metrics,
                **{f"capability_{name}": value for name, value in capability_scores.items()},
            },
            failure_labels=(
                []
                if success
                else list(
                    dict.fromkeys(
                        [
                            "output_mismatch",
                            *(("nondeterministic_output",) if deterministic_rerun != 1.0 else ()),
                            *artifact_failures,
                        ]
                    )
                )
            ),
            diagnostics=diagnostics,
            runtime_seconds=time.perf_counter() - started,
        )
