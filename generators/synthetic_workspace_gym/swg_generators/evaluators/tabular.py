from __future__ import annotations

import json
import time
from pathlib import Path

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.metrics import (
    baseline_normalized_score,
    row_diff_diagnostics,
    row_overlap_metrics,
    weighted_match_score,
)
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class TabularEvaluator(BaseEvaluator):
    def capability_scores(self, expected: object, actual: object) -> dict[str, float]:
        expected_rows = expected if isinstance(expected, list) else []
        actual_rows = actual if isinstance(actual, list) else []
        expected_keys = {frozenset(row) for row in expected_rows if isinstance(row, dict)}
        actual_keys = {frozenset(row) for row in actual_rows if isinstance(row, dict)}
        expected_ids = [str(row.get("account_id")) for row in expected_rows if isinstance(row, dict)]
        actual_ids = [str(row.get("account_id")) for row in actual_rows if isinstance(row, dict)]
        expected_counts = {
            str(row.get("account_id")): row.get("event_count") for row in expected_rows if isinstance(row, dict)
        }
        actual_counts = {
            str(row.get("account_id")): row.get("event_count") for row in actual_rows if isinstance(row, dict)
        }
        expected_amounts = {
            str(row.get("account_id")): row.get("total_amount") for row in expected_rows if isinstance(row, dict)
        }
        actual_amounts = {
            str(row.get("account_id")): row.get("total_amount") for row in actual_rows if isinstance(row, dict)
        }
        return {
            "schema_contract": float(bool(actual_rows) and actual_keys == expected_keys),
            "identity_resolution": float(sorted(actual_ids) == sorted(expected_ids)),
            "filtering_and_deduplication": float(actual_counts == expected_counts),
            "numeric_aggregation": float(actual_amounts == expected_amounts),
            "deterministic_ordering": float(actual_ids == sorted(actual_ids) and bool(actual_ids)),
        }

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
        expected = read_json(hidden_root / "expected_output.json")
        execution = 1.0
        if config.get("entrypoint"):
            precomputed_path = hidden_root / "precomputed_execution.json"
            if not precomputed_path.is_file():
                raise ValueError("tabular execution must be collected in the isolated runner runtime")
            completed = dict(read_json(precomputed_path).get("visible", {}))
            execution = float(int(completed.get("returncode", 1)) == 0)
            if not execution:
                return EvaluatorResult(
                    success=False,
                    score=0.0,
                    subscores={"execution": 0.0, "output_exists": 0.0},
                    failure_labels=["execution_failed"],
                    diagnostics={
                        "stdout": completed.get("stdout", ""),
                        "stderr": completed.get("stderr", ""),
                        "returncode": completed.get("returncode", 1),
                    },
                    runtime_seconds=time.perf_counter() - started,
                )
        output_path = workspace_path / config["output_path"]
        if not output_path.exists():
            return EvaluatorResult(
                success=False,
                score=0.0 if config.get("score_normalization") else 0.2,
                subscores={
                    "output_exists": 0.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["output_missing"],
                diagnostics={"required_output_path": config["output_path"]},
                runtime_seconds=time.perf_counter() - started,
            )
        try:
            actual = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return EvaluatorResult(
                success=False,
                score=0.0 if config.get("score_normalization") else 0.2,
                subscores={
                    "output_exists": 1.0,
                    "valid_json": 0.0,
                    "row_precision": 0.0,
                    "row_recall": 0.0,
                    "row_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["invalid_json"],
                diagnostics={
                    "error": str(exc),
                    "required_output_path": config["output_path"],
                },
                runtime_seconds=time.perf_counter() - started,
            )

        artifact_scores, artifact_failures = self.required_json_artifact_scores(workspace_path, hidden_root, config)
        artifacts_valid = all(score == 1.0 for score in artifact_scores.values())
        success = actual == expected and artifacts_valid
        metrics = row_overlap_metrics(expected, actual)
        score = weighted_match_score(output_exists=1.0, valid_structure=1.0, metrics=metrics)
        capability_scores: dict[str, float] = {}
        if config.get("capability_scoring"):
            capability_scores = self.capability_scores(expected, actual)
            score = sum(capability_scores.values()) / len(capability_scores)
            if (
                capability_scores["filtering_and_deduplication"] == 0.0
                and capability_scores["numeric_aggregation"] == 0.0
            ):
                score = min(score, 0.15)
            elif capability_scores["numeric_aggregation"] == 0.0:
                score = min(score, 0.40)
        if artifact_failures:
            score = min(score, float(config.get("required_artifact_failure_cap", 0.30)))
        raw_score = score
        if config.get("initial_score") is not None:
            score = baseline_normalized_score(raw_score, float(config["initial_score"]))
        if success:
            score = 1.0
        diagnostics = {
            "required_output_path": config["output_path"],
            "expected_rows": len(expected),
            "actual_rows": len(actual) if isinstance(actual, list) else None,
            "raw_score": raw_score,
            "initial_score": config.get("initial_score"),
        }
        if not success:
            diagnostics["expected_preview"] = expected[:2] if isinstance(expected, list) else expected
            diagnostics["actual_preview"] = actual[:2] if isinstance(actual, list) else actual
            diagnostics.update(row_diff_diagnostics(expected, actual))
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={
                "execution": execution,
                "output_exists": 1.0,
                "valid_json": 1.0,
                **metrics,
                **{f"capability_{name}": value for name, value in {**capability_scores, **artifact_scores}.items()},
            },
            failure_labels=([] if success else list(dict.fromkeys(["output_mismatch", *artifact_failures]))),
            diagnostics=diagnostics,
            runtime_seconds=time.perf_counter() - started,
        )
