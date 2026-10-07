from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.evaluators.capabilities import (
    CapabilityScore,
    capability_diagnostics,
    capability_subscores,
    weighted_capability_score,
)
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class TabularCapabilityProgramEvaluator(BaseEvaluator):
    WEIGHTS = {
        "script_exists": 0.03,
        "script_executes": 0.05,
        "valid_json": 0.04,
        "output_schema": 0.05,
        "active_coercion": 0.08,
        "fractional_aggregation": 0.08,
        "canonical_identity": 0.12,
        "deduplication": 0.13,
        "timestamp_normalization": 0.10,
        "temporal_status_join": 0.12,
        "hidden_end_to_end": 0.15,
        "determinism": 0.05,
    }

    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        script_path = workspace_path / str(config["script_path"])
        if not script_path.exists():
            return self._result(started, {"script_exists": 0.0}, ["script_missing"])

        visible = self._run_fixture(
            script_path,
            workspace_path / str(config["visible_input_dir"]),
            timeout=manifest.time_limit_seconds,
        )
        visible_schema = self._schema_score(visible["value"])
        values: dict[str, float] = {
            "script_exists": 1.0,
            "script_executes": float(visible["returncode"] == 0),
            "valid_json": float(visible["value"] is not None),
            "output_schema": visible_schema,
        }

        for item in config.get("focused_fixtures", []):
            fixture = dict(item)
            capability = str(fixture["capability"])
            outcome = self._run_fixture(
                script_path,
                hidden_root / str(fixture["input_dir"]),
                timeout=manifest.time_limit_seconds,
            )
            expected = read_json(hidden_root / str(fixture["expected_path"]))
            values[capability] = float(outcome["value"] == expected)

        hidden_input = hidden_root / str(config["hidden_fixture_dir"])
        first = self._run_fixture(script_path, hidden_input, timeout=manifest.time_limit_seconds)
        second = self._run_fixture(script_path, hidden_input, timeout=manifest.time_limit_seconds)
        hidden_expected = read_json(hidden_root / str(config["hidden_expected_path"]))
        values["hidden_end_to_end"] = float(first["value"] == hidden_expected)
        values["determinism"] = float(
            first["returncode"] == 0
            and second["returncode"] == 0
            and bool(first["bytes"])
            and first["bytes"] == second["bytes"]
        )
        capabilities = [
            CapabilityScore(
                name=name,
                value=values.get(name, 0.0),
                weight=weight,
                diagnostic=("passed" if values.get(name, 0.0) == 1.0 else "failed"),
            )
            for name, weight in self.WEIGHTS.items()
        ]
        success = all(item.clamped_value == 1.0 for item in capabilities)
        score = 1.0 if success else weighted_capability_score(capabilities)
        return EvaluatorResult(
            success=success,
            score=score,
            subscores=capability_subscores(capabilities),
            failure_labels=[] if success else ["program_capabilities_incomplete"],
            diagnostics={
                "capability_diagnostics": capability_diagnostics(capabilities),
                "visible_stderr": visible["stderr"],
                "hidden_stderr": first["stderr"],
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def _run_fixture(self, script_path: Path, input_dir: Path, *, timeout: int) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="swg-tabular-capability-") as tmp:
            root = Path(tmp)
            shutil.copy2(script_path, root / "process_report.py")
            shutil.copytree(input_dir, root / "data")
            output = root / "artifacts" / "report.json"
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(root / "process_report.py"),
                        "--input-dir",
                        "data",
                        "--output",
                        "artifacts/report.json",
                    ],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "returncode": -1,
                    "value": None,
                    "bytes": b"",
                    "stderr": exc.stderr or "timeout",
                }
            payload = output.read_bytes() if output.exists() else b""
            try:
                value = json.loads(payload.decode("utf-8")) if payload else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            return {
                "returncode": completed.returncode,
                "value": value,
                "bytes": payload,
                "stderr": completed.stderr,
            }

    def _schema_score(self, value: object) -> float:
        expected = {"account_id", "event_count", "total_amount"}
        if not isinstance(value, list) or not value:
            return 0.0
        return float(
            all(
                isinstance(row, dict)
                and set(row) == expected
                and isinstance(row["account_id"], str)
                and isinstance(row["event_count"], int)
                and not isinstance(row["event_count"], bool)
                and isinstance(row["total_amount"], (int, float))
                and not isinstance(row["total_amount"], bool)
                for row in value
            )
        )

    def _result(
        self,
        started: float,
        values: dict[str, float],
        labels: list[str],
    ) -> EvaluatorResult:
        capabilities = [
            CapabilityScore(name=name, value=values.get(name, 0.0), weight=weight)
            for name, weight in self.WEIGHTS.items()
        ]
        return EvaluatorResult(
            success=False,
            score=weighted_capability_score(capabilities),
            subscores=capability_subscores(capabilities),
            failure_labels=labels,
            diagnostics={},
            runtime_seconds=time.perf_counter() - started,
        )
