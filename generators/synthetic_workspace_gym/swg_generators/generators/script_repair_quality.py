from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


def enumerate_fix_subsets(labels: Sequence[str]) -> list[frozenset[str]]:
    """Return every meaningful reference-fix subset in stable size/name order."""
    ordered = tuple(dict.fromkeys(str(label) for label in labels))
    return [frozenset(subset) for size in range(len(ordered) + 1) for subset in combinations(ordered, size)]


def validate_partial_solution_lattice(
    scores: Mapping[frozenset[str], float],
    labels: Sequence[str],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Validate that partial fixes improve gradually without dominating D5 reward."""
    ordered = tuple(dict.fromkeys(str(label) for label in labels))
    expected = enumerate_fix_subsets(ordered)
    missing = [sorted(subset) for subset in expected if subset not in scores]
    if missing:
        raise ValueError(f"Missing lattice scores for fix subsets: {missing}")

    limits = {
        "no_fixes_max": 0.15,
        "single_fix_max": 0.40,
        "pair_fix_max": 0.65,
        "all_but_one_max": 0.85,
        "complete_score": 1.0,
        **dict(thresholds or {}),
    }
    by_size: dict[int, list[float]] = {}
    for subset in expected:
        by_size.setdefault(len(subset), []).append(float(scores[subset]))

    violations: list[str] = []
    if float(scores[frozenset()]) > float(limits["no_fixes_max"]):
        violations.append("untouched workspace reward exceeds no_fixes_max")
    if by_size.get(1) and max(by_size[1]) > float(limits["single_fix_max"]):
        violations.append("a single fix exceeds single_fix_max")
    if len(ordered) > 2 and by_size.get(2) and max(by_size[2]) > float(limits["pair_fix_max"]):
        violations.append("a two-fix subset exceeds pair_fix_max")
    if len(ordered) > 2 and max(by_size[len(ordered) - 1]) > float(limits["all_but_one_max"]):
        violations.append("an all-but-one subset exceeds all_but_one_max")
    full_score = float(scores[frozenset(ordered)])
    if abs(full_score - float(limits["complete_score"])) > 1e-9:
        violations.append("complete reference solution does not reach complete_score")

    profile = {
        "bug_count": len(ordered),
        "subset_count": len(expected),
        "no_fix_score": round(float(scores[frozenset()]), 6),
        "single_fix_max_score": round(max(by_size.get(1, [0.0])), 6),
        "pair_fix_max_score": round(max(by_size.get(2, [0.0])), 6),
        "all_but_one_max_score": round(max(by_size.get(max(0, len(ordered) - 1), [0.0])), 6),
        "full_solution_score": round(full_score, 6),
        "valid": not violations,
        "violations": violations,
    }
    if violations:
        raise ValueError("; ".join(violations))
    return profile


def evaluate_fix_lattice(
    *,
    correct_files: Mapping[str, str],
    selected_bugs: Sequence[Mapping[str, Any]],
    visible_template: Path,
    scratch_root: Path,
    evaluator: Any,
    manifest: Any,
    hidden_root: Path,
) -> dict[frozenset[str], float]:
    """Evaluate every meaningful subset of reference fixes in isolated workspaces."""
    labels = [str(bug["label"]) for bug in selected_bugs]
    scores: dict[frozenset[str], float] = {}
    for index, fixed in enumerate(enumerate_fix_subsets(labels)):
        workspace = scratch_root / f"subset-{index:03d}"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(visible_template, workspace)
        for relative_path, content in correct_files.items():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        for bug in selected_bugs:
            if str(bug["label"]) in fixed:
                continue
            target = workspace / str(bug["target_path"])
            target.write_text(
                bug["apply"](target.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        scores[fixed] = float(evaluator.evaluate(workspace, manifest, hidden_root).score)
    return scores
