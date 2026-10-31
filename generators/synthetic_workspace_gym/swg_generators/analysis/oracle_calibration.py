from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from swg_generators.analysis.calibration import build_monotonicity_report
from swg_generators.evaluators.base import BaseEvaluator
from swg_generators.schemas import EnvironmentManifest


def evaluate_ordered_oracle_states(
    *,
    task_id: str,
    correct_files: Mapping[str, str],
    selected_bugs: Sequence[Mapping[str, object]],
    visible_template: Path,
    scratch_root: Path,
    evaluator: BaseEvaluator,
    manifest: EnvironmentManifest,
    hidden_root: Path,
    capability_count: int,
    semantic_dependency_depth: int,
) -> dict[str, object]:
    """Evaluate five useful reference states without enumerating a full lattice."""

    labels = [str(bug["label"]) for bug in selected_bugs]
    if len(labels) == 2:
        fixed_sets = [
            ("untouched", frozenset()),
            ("one_capability", frozenset(labels[:1])),
            ("all_but_one", frozenset(labels[:-1])),
            ("two_capabilities", frozenset(labels[:2])),
            ("full", frozenset(labels)),
        ]
    else:
        fixed_sets = [
            ("untouched", frozenset()),
            ("one_capability", frozenset(labels[:1])),
            ("two_capabilities", frozenset(labels[:2])),
            ("all_but_one", frozenset(labels[:-1])),
            ("full", frozenset(labels)),
        ]
    ordered_scores: list[tuple[str, float]] = []
    for index, (state_name, fixed) in enumerate(fixed_sets):
        workspace = scratch_root / f"{index:02d}-{state_name}"
        shutil.copytree(visible_template, workspace)
        candidate_files = dict(correct_files)
        for bug in selected_bugs:
            if str(bug["label"]) in fixed:
                continue
            target_path = str(bug["target_path"])
            candidate_files[target_path] = bug["apply"](candidate_files[target_path])
        for relative_path, content in candidate_files.items():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        result = evaluator.evaluate(workspace, manifest, hidden_root)
        ordered_scores.append((state_name, float(result.score)))
    return build_monotonicity_report(
        task_id=task_id,
        ordered_states=ordered_scores,
        capability_count=capability_count,
        semantic_dependency_depth=semantic_dependency_depth,
    )
