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


class ProfiledPipelineEvaluator(BaseEvaluator):
    WEIGHTS = {
        "execution": 0.05,
        "top_level_shape": 0.05,
        "row_schema": 0.10,
        "normalization": 0.15,
        "deduplication": 0.15,
        "filtering": 0.10,
        "aggregation": 0.25,
        "ordering": 0.05,
        "determinism": 0.10,
    }

    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        started = time.perf_counter()
        workspace_path = workspace_path.resolve()
        hidden_root = hidden_root.resolve()
        config = read_json(hidden_root / "evaluator_config.json")
        visible = self._run_workspace(
            workspace_path,
            entrypoint=str(config["entrypoint"]),
            output_path=str(config["required_output_path"]),
            timeout=manifest.time_limit_seconds,
            rerun=True,
        )
        if visible["returncode"] != 0:
            return self._hard_failure(started, "execution_failed", visible)
        if not visible["exists"]:
            return self._hard_failure(started, "output_missing", visible)
        if visible["value"] is None:
            return self._hard_failure(started, "invalid_json", visible, score=0.10)

        with tempfile.TemporaryDirectory(prefix="swg-pipeline-hidden-") as tmp:
            hidden_workspace = Path(tmp) / "workspace"
            shutil.copytree(workspace_path, hidden_workspace)
            fixture_root = hidden_root / str(config["hidden_fixture_dir"])
            for source in fixture_root.rglob("*"):
                if not source.is_file():
                    continue
                target = hidden_workspace / source.relative_to(fixture_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            shutil.rmtree(hidden_workspace / "artifacts", ignore_errors=True)
            hidden = self._run_workspace(
                hidden_workspace,
                entrypoint=str(config["entrypoint"]),
                output_path=str(config["required_output_path"]),
                timeout=manifest.time_limit_seconds,
                rerun=True,
            )

        visible_expected = read_json(hidden_root / "expected_output.json")
        hidden_expected = read_json(hidden_root / str(config["hidden_expected_path"]))
        visible_scores = self._semantic_scores(visible_expected, visible["value"])
        hidden_scores = self._semantic_scores(hidden_expected, hidden["value"])
        values = {
            "execution": float(hidden["returncode"] == 0),
            **{name: round((visible_scores[name] + hidden_scores[name]) / 2, 6) for name in visible_scores},
            "determinism": min(float(visible["deterministic"]), float(hidden["deterministic"])),
        }
        capabilities = [
            CapabilityScore(
                name=name,
                value=values.get(name, 0.0),
                weight=weight,
                diagnostic=("passed" if values.get(name, 0.0) == 1.0 else "partial or failed"),
            )
            for name, weight in self.WEIGHTS.items()
        ]
        success = bool(
            visible["value"] == visible_expected and hidden["value"] == hidden_expected and values["determinism"] == 1.0
        )
        base_score = weighted_capability_score(capabilities)
        # D5 should reward completion of the semantic chain, not the many
        # mechanical properties (execution, JSON shape, determinism) that a
        # substantially broken starter already satisfies.  A steep completion
        # factor keeps untouched and one-defect repairs below the shared D5
        # ceilings while preserving continuous partial credit.
        semantic_names = {
            "normalization",
            "deduplication",
            "filtering",
            "aggregation",
            "ordering",
        }
        semantic_completion = sum(values[name] for name in semantic_names) / len(semantic_names)
        score = 1.0 if success else round(base_score * semantic_completion**6, 6)
        return EvaluatorResult(
            success=success,
            score=score,
            subscores=capability_subscores(capabilities),
            failure_labels=[] if success else ["pipeline_capabilities_incomplete"],
            diagnostics={
                "capability_diagnostics": capability_diagnostics(capabilities),
                "visible_exact": visible["value"] == visible_expected,
                "hidden_exact": hidden["value"] == hidden_expected,
                "base_weighted_score": base_score,
                "semantic_completion": round(semantic_completion, 6),
                "visible_stderr": visible["stderr"],
                "hidden_stderr": hidden["stderr"],
            },
            runtime_seconds=time.perf_counter() - started,
        )

    def _semantic_scores(self, expected: object, actual: object) -> dict[str, float]:
        expected_rows = expected if isinstance(expected, list) else []
        actual_rows = actual if isinstance(actual, list) else []
        shape = float(isinstance(actual, list))
        keys = {"team", "job_count", "total_hours"}
        schema = float(bool(actual_rows) and all(isinstance(row, dict) and set(row) == keys for row in actual_rows))
        expected_by_team = {str(row["team"]): row for row in expected_rows if isinstance(row, dict)}
        actual_by_team = {str(row["team"]): row for row in actual_rows if isinstance(row, dict)}
        expected_teams = set(expected_by_team)
        actual_teams = set(actual_by_team)
        team_union = expected_teams | actual_teams
        normalization = len(expected_teams & actual_teams) / len(team_union) if team_union else 1.0
        expected_count = sum(int(row["job_count"]) for row in expected_by_team.values())
        actual_count = sum(int(row.get("job_count", 0)) for row in actual_by_team.values() if isinstance(row, dict))
        count_score = 1.0 - min(1.0, abs(expected_count - actual_count) / max(1, expected_count))
        aggregation = sum(
            actual_by_team.get(team, {}).get("total_hours") == row.get("total_hours")
            for team, row in expected_by_team.items()
        ) / max(1, len(expected_by_team))
        actual_order = [str(row.get("team")) for row in actual_rows if isinstance(row, dict)]
        return {
            "top_level_shape": shape,
            "row_schema": schema,
            "normalization": round(normalization, 6),
            "deduplication": round(count_score, 6),
            "filtering": round(count_score, 6),
            "aggregation": round(aggregation, 6),
            "ordering": float(actual_order == sorted(actual_order) and bool(actual_order)),
        }

    def _run_workspace(
        self,
        root: Path,
        *,
        entrypoint: str,
        output_path: str,
        timeout: int,
        rerun: bool,
    ) -> dict[str, object]:
        command = [sys.executable, str(root / entrypoint)]
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = root / output_path
        first = output.read_bytes() if output.exists() else b""
        second_completed = completed
        if rerun and completed.returncode == 0:
            second_completed = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        second = output.read_bytes() if output.exists() else b""
        try:
            value = json.loads(first.decode("utf-8")) if first else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        return {
            "returncode": completed.returncode,
            "exists": bool(first),
            "value": value,
            "deterministic": bool(
                first and completed.returncode == 0 and second_completed.returncode == 0 and first == second
            ),
            "stderr": completed.stderr,
        }

    def _hard_failure(
        self,
        started: float,
        label: str,
        outcome: dict[str, object],
        *,
        score: float = 0.0,
    ) -> EvaluatorResult:
        return EvaluatorResult(
            success=False,
            score=score,
            subscores={},
            failure_labels=[label],
            diagnostics={"stderr": outcome.get("stderr", "")},
            runtime_seconds=time.perf_counter() - started,
        )
