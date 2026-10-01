from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.capabilities import (
    CapabilityScore,
    capability_diagnostics,
    capability_subscores,
    weighted_capability_score,
)
from swg_generators.evaluators.metrics import (
    baseline_normalized_score,
    flatten_json,
    json_field_diff_diagnostics,
    row_overlap_metrics,
    weighted_match_score,
)
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class RetrievalWorkspaceEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        mode = str(config.get("mode", "exact_json"))
        if mode == "exact_json":
            return self.evaluate_exact_json(workspace_path, manifest, hidden_root, config)
        if mode == "hidden_tests":
            return self.evaluate_hidden_tests(workspace_path, manifest, hidden_root, config)
        raise ValueError(f"Unsupported retrieval workspace evaluation mode: {mode}")

    def evaluate_exact_json(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        hidden_root: Path,
        config: dict[str, object],
    ) -> EvaluatorResult:
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
                diagnostics={
                    "error": str(exc),
                    "required_output_path": str(config["output_path"]),
                },
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

    def evaluate_hidden_tests(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        hidden_root: Path,
        config: dict[str, object],
    ) -> EvaluatorResult:
        started = time.perf_counter()
        target_path = config.get("target_path")
        if target_path and not (workspace_path / str(target_path)).exists():
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={},
                failure_labels=["target_missing"],
                diagnostics={"target_path": str(target_path)},
                runtime_seconds=time.perf_counter() - started,
            )
        runner_path = (hidden_root / str(config["runner"])).resolve()
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(workspace_path)],
                cwd=str(hidden_root.resolve()),
                capture_output=True,
                text=True,
                timeout=manifest.time_limit_seconds,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluatorResult(
                success=False,
                score=0.0,
                subscores={},
                failure_labels=["timeout"],
                diagnostics={
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "runner": str(runner_path),
                },
                runtime_seconds=time.perf_counter() - started,
            )

        payload = self.extract_payload(completed.stdout)
        if payload is None:
            return EvaluatorResult(
                success=False,
                score=0.05,
                subscores={},
                failure_labels=["import_or_execution_failed"],
                diagnostics={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                runtime_seconds=time.perf_counter() - started,
            )

        raw_subscores = {str(name): float(value) for name, value in dict(payload.get("subscores", {})).items()}
        raw_weights = {str(name): float(value) for name, value in dict(payload.get("capability_weights", {})).items()}
        capabilities: list[CapabilityScore] = []
        for raw_name, value in raw_subscores.items():
            if not raw_name.startswith("capability_"):
                continue
            name = raw_name.removeprefix("capability_")
            capabilities.append(
                CapabilityScore(
                    name=name,
                    value=value,
                    weight=raw_weights.get(name, 1.0),
                    diagnostic=("passed" if value >= 1.0 else "failed or incomplete"),
                )
            )

        tests_passed = float(raw_subscores.get("tests_passed", 0.0))
        tests_total = float(raw_subscores.get("tests_total", 0.0))
        tests_passed_ratio = (tests_passed / tests_total) if tests_total else 0.0
        artifact_scores, artifact_failures = self.required_json_artifact_scores(workspace_path, hidden_root, config)
        artifact_weights = {
            str(name): float(value) for name, value in dict(config.get("required_json_artifact_weights", {})).items()
        }
        for name, value in artifact_scores.items():
            capabilities.append(
                CapabilityScore(
                    name=name,
                    value=value,
                    weight=artifact_weights.get(name, 0.10),
                    diagnostic=(
                        "required artifact matches" if value == 1.0 else "required artifact missing or mismatched"
                    ),
                )
            )
        raw_score = weighted_capability_score(capabilities) if capabilities else round(tests_passed_ratio, 6)
        baseline_score = config.get("initial_score")
        score = baseline_normalized_score(raw_score, float(baseline_score)) if baseline_score is not None else raw_score
        artifacts_valid = all(value == 1.0 for value in artifact_scores.values())
        success = bool(payload.get("success")) and artifacts_valid
        if success:
            score = 1.0
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "tests_passed_ratio": round(tests_passed_ratio, 6),
                **capability_subscores(capabilities),
            },
            failure_labels=list(dict.fromkeys([*list(payload.get("failure_labels", [])), *artifact_failures])),
            diagnostics={
                **dict(payload.get("diagnostics", {})),
                "capability_diagnostics": capability_diagnostics(capabilities),
                "artifact_failures": artifact_failures,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "raw_score": raw_score,
                "initial_score": baseline_score,
            },
            runtime_seconds=time.perf_counter() - started,
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
            actual_path = workspace_path / str(item["path"])
            expected_path = hidden_root / str(item["expected_path"])
            score = 0.0
            try:
                actual = json.loads(actual_path.read_text(encoding="utf-8"))
                expected = read_json(expected_path)
                score = 1.0 if actual == expected else 0.0
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                score = 0.0
            scores[capability] = score
            if score == 0.0:
                failures.append(f"{capability}_failed")
        return scores, failures

    def field_overlap_metrics(self, expected: Any, actual: Any) -> dict[str, float]:
        row_metrics = row_overlap_metrics(self.flatten_json(expected), self.flatten_json(actual))
        return {
            "field_precision": row_metrics["row_precision"],
            "field_recall": row_metrics["row_recall"],
            "field_f1": row_metrics["row_f1"],
            "exact_match": row_metrics["exact_match"],
        }

    def flatten_json(self, value: Any, *, prefix: str = "$") -> list[dict[str, object]]:
        return flatten_json(value, prefix=prefix)

    def preview(self, value: Any) -> Any:
        if isinstance(value, list):
            return value[:2]
        if isinstance(value, dict):
            keys = sorted(value)[:4]
            return {key: value[key] for key in keys}
        return value

    def extract_payload(self, stdout: str) -> dict[str, object] | None:
        for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
