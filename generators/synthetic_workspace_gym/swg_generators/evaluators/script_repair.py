from __future__ import annotations

import json
import time
from pathlib import Path

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.metrics import baseline_normalized_score
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class ScriptRepairEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        del manifest
        started = time.perf_counter()
        config = read_json(hidden_root / "evaluator_config.json")
        if config.get("secure_protocol") != "observations-v1":
            raise ValueError("script-repair tasks require the secure observations protocol")
        return self.evaluate_observations(workspace_path, hidden_root, config, started)

    def evaluate_observations(
        self,
        workspace_path: Path,
        hidden_root: Path,
        config: dict[str, object],
        started: float,
    ) -> EvaluatorResult:
        execution = read_json(hidden_root / "precomputed_execution.json")
        expected = read_json(hidden_root / str(config["expected_observations_path"]))
        visible_execution = dict(execution.get("visible", {}))
        observations_path = hidden_root / "precomputed_outputs" / "observations.json"
        actual = None
        if int(visible_execution.get("returncode", 1)) == 0 and bool(execution.get("observations_present", False)):
            try:
                actual = json.loads(observations_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

        outcomes = self._compare_observations(expected, actual)
        tests_total = len(outcomes)
        tests_passed = sum(outcomes.values())
        ratio = tests_passed / tests_total if tests_total else 0.0
        artifact_scores, artifact_failures = self.required_json_artifact_scores(workspace_path, hidden_root, config)
        capability_scores = self._observation_capabilities(outcomes, config)
        if capability_scores:
            capability_scores.update(artifact_scores)
        success = bool(tests_total and tests_passed == tests_total and not artifact_failures)
        score = 1.0 if success else ratio
        if capability_scores:
            score = sum(capability_scores.values()) / len(capability_scores)
        if artifact_failures:
            score = min(score, float(config.get("required_artifact_failure_cap", 0.30)))
        raw_score = score
        if config.get("initial_score") is not None:
            score = baseline_normalized_score(raw_score, float(config["initial_score"]))
        if success:
            score = 1.0
        return EvaluatorResult(
            success=success,
            score=round(score, 6),
            subscores={
                "tests_passed": tests_passed,
                "tests_total": float(tests_total),
                "tests_passed_ratio": round(ratio, 6),
                **{f"capability_{name}": round(value, 6) for name, value in capability_scores.items()},
            },
            failure_labels=[] if success else ["hidden_observations_mismatch", *artifact_failures],
            diagnostics={
                "failed_observations": [name for name, value in outcomes.items() if value == 0.0],
                "execution": execution,
                "artifact_failures": artifact_failures,
                "raw_score": raw_score,
                "initial_score": config.get("initial_score"),
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def _compare_observations(self, expected: object, actual: object) -> dict[str, float]:
        if isinstance(expected, list):
            actual_by_name = {
                str(item.get("name")): item.get("value") for item in actual or [] if isinstance(item, dict)
            }
            return {
                str(item["name"]): float(actual_by_name.get(str(item["name"])) == item.get("value"))
                for item in expected
                if isinstance(item, dict) and "name" in item
            }
        if isinstance(expected, dict):
            actual_values = actual if isinstance(actual, dict) else {}
            return {name: float(actual_values.get(name) == value) for name, value in expected.items()}
        raise TypeError("expected observations must be a list or mapping")

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

    def _observation_capabilities(
        self,
        outcomes: dict[str, float],
        config: dict[str, object],
    ) -> dict[str, float]:
        groups = dict(config.get("capability_groups", {}))
        if not groups:
            return {}
        scores: dict[str, float] = {}
        for capability, methods_value in groups.items():
            methods = [str(method) for method in methods_value]
            relevant = [
                value
                for name, value in outcomes.items()
                if any(name.startswith(f"test_{method}:") or name == method for method in methods)
            ]
            if relevant:
                scores[str(capability)] = sum(relevant) / len(relevant)
        return scores
