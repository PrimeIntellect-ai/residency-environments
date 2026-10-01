from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.metrics import baseline_normalized_score
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class ScriptRepairEvaluator(BaseEvaluator):
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        runner_path = (hidden_root / config["runner"]).resolve()
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
                subscores={
                    "tests_passed": 0.0,
                    "tests_total": 0.0,
                    "tests_passed_ratio": 0.0,
                },
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
                score=0.0,
                subscores={
                    "tests_passed": 0.0,
                    "tests_total": 0.0,
                    "tests_passed_ratio": 0.0,
                },
                failure_labels=["hidden_tests_failed"],
                diagnostics={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                runtime_seconds=time.perf_counter() - started,
            )

        tests_passed = float(payload["subscores"].get("tests_passed", 0))
        tests_total = float(payload["subscores"].get("tests_total", 0))
        tests_passed_ratio = (tests_passed / tests_total) if tests_total else 0.0
        capability_scores = self.capability_scores(payload, config)
        artifact_scores, artifact_failures = self.required_json_artifact_scores(workspace_path, hidden_root, config)
        capability_scores.update(artifact_scores)
        public_execution, public_diagnostics = self.evaluate_public_entrypoint(workspace_path, manifest, config)
        if config.get("public_entrypoint"):
            capability_scores["public_execution"] = public_execution
        score = tests_passed_ratio
        if capability_scores:
            score = sum(capability_scores.values()) / len(capability_scores)
            for capability, cap in dict(config.get("capability_score_caps", {})).items():
                if capability_scores.get(str(capability), 0.0) == 0.0:
                    score = min(score, float(cap))
        if artifact_failures or public_execution == 0.0:
            score = min(score, float(config.get("required_artifact_failure_cap", 0.30)))
        raw_score = score
        if config.get("initial_score") is not None:
            score = baseline_normalized_score(raw_score, float(config["initial_score"]))
        success = bool(payload["success"]) and not artifact_failures and public_execution == 1.0
        if success:
            score = 1.0
        return EvaluatorResult(
            success=success,
            score=round(score, 6),
            subscores={
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "tests_passed_ratio": round(tests_passed_ratio, 6),
                **{f"capability_{name}": round(value, 6) for name, value in capability_scores.items()},
            },
            failure_labels=list(
                dict.fromkeys(
                    [
                        *list(payload.get("failure_labels", [])),
                        *artifact_failures,
                        *(["public_entrypoint_failed"] if public_execution == 0.0 else []),
                    ]
                )
            ),
            diagnostics={
                **payload.get("diagnostics", {}),
                **public_diagnostics,
                "artifact_failures": artifact_failures,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "raw_score": raw_score,
                "initial_score": config.get("initial_score"),
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

    def evaluate_public_entrypoint(
        self,
        workspace_path: Path,
        manifest: EnvironmentManifest,
        config: dict[str, object],
    ) -> tuple[float, dict[str, object]]:
        entrypoint = config.get("public_entrypoint")
        if not entrypoint:
            return 1.0, {}
        try:
            completed = subprocess.run(
                [sys.executable, str(workspace_path / str(entrypoint))],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=manifest.time_limit_seconds,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            return 0.0, {
                "public_stdout": exc.stdout or "",
                "public_stderr": exc.stderr or "",
                "public_timeout": True,
            }
        return float(completed.returncode == 0), {
            "public_stdout": completed.stdout,
            "public_stderr": completed.stderr,
            "public_returncode": completed.returncode,
        }

    def capability_scores(self, payload: dict[str, object], config: dict[str, object]) -> dict[str, float]:
        groups = dict(config.get("capability_groups", {}))
        if not groups:
            return {}
        diagnostics = dict(payload.get("diagnostics", {}))
        failed_ids = {
            str(item)
            for item in [
                *list(diagnostics.get("failures", [])),
                *list(diagnostics.get("errors", [])),
            ]
        }
        scores: dict[str, float] = {}
        for capability, methods_value in groups.items():
            methods = [str(method) for method in methods_value]
            if not methods:
                continue
            failed = sum(any(test_id.endswith(f".{method}") for test_id in failed_ids) for method in methods)
            scores[str(capability)] = (len(methods) - failed) / len(methods)
        return scores

    def extract_payload(self, stdout: str) -> dict[str, object] | None:
        for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
