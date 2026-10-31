from __future__ import annotations

import json
import time
from pathlib import Path

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.evaluators.capabilities import (
    CapabilityScore,
    capability_diagnostics,
    capability_subscores,
    weighted_capability_score,
)
from synthetic_workspace_gym.schemas import EnvironmentManifest, EvaluatorResult
from synthetic_workspace_gym.utils.io import read_json


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
        del manifest
        started = time.perf_counter()
        config = read_json(hidden_root / "evaluator_config.json")
        execution_path = hidden_root / "precomputed_execution.json"
        if not execution_path.is_file():
            raise ValueError("profiled pipeline execution must run in the isolated runner runtime")
        execution = read_json(execution_path)
        visible = self._precomputed_result(
            hidden_root / "precomputed_outputs" / "visible-output.json",
            dict(execution.get("visible", {})),
        )
        hidden = self._precomputed_result(
            hidden_root / "precomputed_outputs" / "hidden-output.json",
            dict(execution.get("hidden", {})),
        )
        if visible["returncode"] != 0:
            return self._hard_failure(started, "execution_failed", visible)
        if not visible["exists"]:
            return self._hard_failure(started, "output_missing", visible)
        if visible["value"] is None:
            return self._hard_failure(started, "invalid_json", visible, score=0.10)

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
                diagnostic="passed" if values.get(name, 0.0) == 1.0 else "partial or failed",
            )
            for name, weight in self.WEIGHTS.items()
        ]
        success = bool(
            visible["value"] == visible_expected and hidden["value"] == hidden_expected and values["determinism"] == 1.0
        )
        base_score = weighted_capability_score(capabilities)
        semantic_names = {"normalization", "deduplication", "filtering", "aggregation", "ordering"}
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
        actual_count = sum(int(row.get("job_count", 0)) for row in actual_by_team.values())
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

    def _precomputed_result(self, output: Path, run: dict[str, object]) -> dict[str, object]:
        raw = output.read_bytes() if output.is_file() else b""
        try:
            value = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            value = None
        return {
            "returncode": int(run.get("returncode", 1)),
            "exists": bool(raw),
            "value": value,
            "deterministic": bool(run.get("deterministic", False)),
            "stderr": str(run.get("stderr", "")),
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
            subscores={name: 0.0 for name in self.WEIGHTS},
            failure_labels=[label],
            diagnostics={"returncode": outcome["returncode"], "stderr": outcome["stderr"]},
            runtime_seconds=time.perf_counter() - started,
        )
