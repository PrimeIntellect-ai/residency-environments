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
from swg_generators.evaluators.metrics import row_overlap_metrics
from swg_generators.schemas import EnvironmentManifest, EvaluatorResult
from swg_generators.utils.io import read_json


class TabularProgramEvaluator(BaseEvaluator):
    WEIGHTS = {
        "visible_output": 0.20,
        "hidden_fixture_semantics": 0.45,
        "output_schema": 0.10,
        "operation_order_edge_cases": 0.15,
        "determinism": 0.10,
    }

    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        script_path = workspace_path / str(config["script_path"])
        if not script_path.exists():
            return self._failure(started, "script_missing", {"script_exists": 0.0})

        visible_output = workspace_path / str(config["output_path"])
        visible_run = self._run_script(
            script_path,
            cwd=workspace_path,
            input_dir=str(config["visible_input_dir"]),
            output_path=str(config["output_path"]),
            timeout=manifest.time_limit_seconds,
        )
        if visible_run.returncode != 0:
            return self._failure(
                started,
                "execution_failed",
                {"execution": 0.0},
                stdout=visible_run.stdout,
                stderr=visible_run.stderr,
            )
        visible_actual = self._read_json(visible_output)
        if visible_actual is None:
            return self._failure(started, "visible_output_invalid", {"execution": 1.0})
        visible_expected = read_json(hidden_root / "expected_output.json")

        with tempfile.TemporaryDirectory(prefix="swg-tabular-hidden-") as tmp:
            hidden_workspace = Path(tmp)
            shutil.copy2(script_path, hidden_workspace / "process_report.py")
            shutil.copytree(
                hidden_root / str(config["hidden_fixture_dir"]),
                hidden_workspace / "data",
            )
            hidden_output = hidden_workspace / "artifacts" / "report.json"
            first = self._run_script(
                hidden_workspace / "process_report.py",
                cwd=hidden_workspace,
                input_dir="data",
                output_path="artifacts/report.json",
                timeout=manifest.time_limit_seconds,
            )
            hidden_actual = self._read_json(hidden_output) if first.returncode == 0 else None
            first_bytes = hidden_output.read_bytes() if hidden_output.exists() else b""
            second = self._run_script(
                hidden_workspace / "process_report.py",
                cwd=hidden_workspace,
                input_dir="data",
                output_path="artifacts/report.json",
                timeout=manifest.time_limit_seconds,
            )
            second_bytes = hidden_output.read_bytes() if hidden_output.exists() else b""

        hidden_expected = read_json(hidden_root / str(config["hidden_expected_path"]))
        visible_exact = float(visible_actual == visible_expected)
        hidden_metrics = row_overlap_metrics(hidden_expected, hidden_actual or [])
        hidden_semantics = float(hidden_metrics["row_f1"])
        schema = 0.5 * (self._schema_score(visible_actual) + self._schema_score(hidden_actual))
        edge_score = self._edge_score(
            hidden_expected,
            hidden_actual,
            [str(value) for value in config.get("edge_account_ids", [])],
        )
        hidden_exact = hidden_actual == hidden_expected
        determinism = float(
            hidden_exact
            and first.returncode == 0
            and second.returncode == 0
            and bool(first_bytes)
            and first_bytes == second_bytes
        )
        capabilities = {
            "visible_output": visible_exact,
            "hidden_fixture_semantics": hidden_semantics,
            "output_schema": schema,
            "operation_order_edge_cases": edge_score,
            "determinism": determinism,
        }
        score = round(
            sum(self.WEIGHTS[name] * capabilities[name] for name in self.WEIGHTS),
            6,
        )
        success = bool(visible_exact == 1.0 and hidden_exact and schema == 1.0 and determinism == 1.0)
        if success:
            score = 1.0
        return EvaluatorResult(
            success=success,
            score=score,
            subscores={
                "execution": 1.0,
                **{f"capability_{name}": value for name, value in capabilities.items()},
                **{f"hidden_{name}": value for name, value in hidden_metrics.items()},
            },
            failure_labels=[] if success else ["program_semantics_mismatch"],
            diagnostics={
                "visible_exact": bool(visible_exact),
                "hidden_exact": bool(hidden_exact),
                "hidden_first_returncode": first.returncode,
                "hidden_second_returncode": second.returncode,
                "hidden_stdout": first.stdout,
                "hidden_stderr": first.stderr,
                "visible_preview": self._preview(visible_actual),
                "hidden_preview": self._preview(hidden_actual),
                "expected_hidden_preview": self._preview(hidden_expected),
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def _run_script(
        self,
        script_path: Path,
        *,
        cwd: Path,
        input_dir: str,
        output_path: str,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-dir",
                input_dir,
                "--output",
                output_path,
            ],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def _schema_score(self, value: object) -> float:
        if not isinstance(value, list) or not value:
            return 0.0
        expected_keys = {"account_id", "event_count", "total_amount"}
        return float(
            all(
                isinstance(row, dict)
                and set(row) == expected_keys
                and isinstance(row["account_id"], str)
                and isinstance(row["event_count"], int)
                and not isinstance(row["event_count"], bool)
                and isinstance(row["total_amount"], (int, float))
                and not isinstance(row["total_amount"], bool)
                for row in value
            )
        )

    def _edge_score(
        self,
        expected: object,
        actual: object,
        edge_account_ids: list[str],
    ) -> float:
        if not edge_account_ids or not isinstance(expected, list) or not isinstance(actual, list):
            return 0.0
        expected_by_id = {str(row.get("account_id")): row for row in expected if isinstance(row, dict)}
        actual_by_id = {str(row.get("account_id")): row for row in actual if isinstance(row, dict)}
        passed = sum(actual_by_id.get(account_id) == expected_by_id.get(account_id) for account_id in edge_account_ids)
        return passed / len(edge_account_ids)

    def _read_json(self, path: Path) -> object | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _preview(self, value: object) -> object:
        return value[:2] if isinstance(value, list) else value

    def _failure(
        self,
        started: float,
        label: str,
        subscores: dict[str, float],
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> EvaluatorResult:
        return EvaluatorResult(
            success=False,
            score=0.0,
            subscores=subscores,
            failure_labels=[label],
            diagnostics={"stdout": stdout, "stderr": stderr},
            runtime_seconds=time.perf_counter() - started,
        )
