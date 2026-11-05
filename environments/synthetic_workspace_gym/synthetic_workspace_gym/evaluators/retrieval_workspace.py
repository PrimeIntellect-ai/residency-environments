from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.evaluators.metrics import (
    baseline_normalized_score,
    flatten_json,
    json_field_diff_diagnostics,
    row_overlap_metrics,
    weighted_match_score,
)
from synthetic_workspace_gym.schemas import EnvironmentManifest, EvaluatorResult
from synthetic_workspace_gym.utils.io import read_json

from .script_repair import ScriptRepairEvaluator


class RetrievalWorkspaceEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        if config.get("secure_protocol") == "observations-v1":
            return ScriptRepairEvaluator().evaluate_observations(
                workspace_path,
                hidden_root,
                config,
                time.perf_counter(),
            )
        if str(config.get("mode", "exact_json")) != "exact_json":
            raise ValueError("retrieval hidden tests require the secure observations protocol")
        return self.evaluate_exact_json(workspace_path, manifest, hidden_root, config)

    def evaluate_exact_json(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        hidden_root: Path,
        config: dict[str, object],
    ) -> EvaluatorResult:
        del manifest
        started = time.perf_counter()
        output_path = workspace_path / str(config["output_path"])
        if not output_path.exists():
            return EvaluatorResult(
                success=False,
                score=0.0 if config.get("score_normalization") else 0.2,
                subscores={
                    "output_exists": 0.0,
                    "valid_json": 0.0,
                    "field_precision": 0.0,
                    "field_recall": 0.0,
                    "field_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["output_missing"],
                diagnostics={"required_output_path": str(config["output_path"])},
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
                    "field_precision": 0.0,
                    "field_recall": 0.0,
                    "field_f1": 0.0,
                    "exact_match": 0.0,
                },
                failure_labels=["invalid_json"],
                diagnostics={"error": str(exc), "required_output_path": str(config["output_path"])},
                runtime_seconds=time.perf_counter() - started,
            )

        expected = read_json(hidden_root / str(config.get("expected_path", "expected_output.json")))
        metrics = self.field_overlap_metrics(expected, actual)
        raw_score = weighted_match_score(
            output_exists=1.0,
            valid_structure=1.0,
            metrics={
                "row_precision": metrics["field_precision"],
                "row_recall": metrics["field_recall"],
                "row_f1": metrics["field_f1"],
                "exact_match": metrics["exact_match"],
            },
        )
        baseline_score = config.get("initial_score")
        score = baseline_normalized_score(raw_score, float(baseline_score)) if baseline_score is not None else raw_score
        success = actual == expected
        diagnostics = {
            "required_output_path": str(config["output_path"]),
            "expected_preview": self.preview(expected),
            "actual_preview": self.preview(actual),
            "raw_score": raw_score,
            "initial_score": baseline_score,
        }
        if not success:
            diagnostics.update(json_field_diff_diagnostics(expected, actual))
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={"output_exists": 1.0, "valid_json": 1.0, **metrics},
            failure_labels=[] if success else ["output_mismatch"],
            diagnostics=diagnostics,
            runtime_seconds=time.perf_counter() - started,
        )

    def field_overlap_metrics(self, expected: Any, actual: Any) -> dict[str, float]:
        row_metrics = row_overlap_metrics(flatten_json(expected), flatten_json(actual))
        return {
            "field_precision": row_metrics["row_precision"],
            "field_recall": row_metrics["row_recall"],
            "field_f1": row_metrics["row_f1"],
            "exact_match": row_metrics["exact_match"],
        }

    def preview(self, value: Any) -> Any:
        if isinstance(value, list):
            return value[:2]
        if isinstance(value, dict):
            keys = sorted(value)[:4]
            return {key: value[key] for key in keys}
        return value
