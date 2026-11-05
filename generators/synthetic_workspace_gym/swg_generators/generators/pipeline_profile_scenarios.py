from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from textwrap import dedent

from swg_generators.generators.d5_profiles import select_weighted_d5_profile
from swg_generators.schemas import EnvironmentSpec


def build_profiled_team_hours_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    profile = select_weighted_d5_profile(spec.difficulty, spec.seed)
    if profile is None:
        raise ValueError("profiled team-hours scenarios require difficulty 5")
    visible = _fixture(rng, profile.profile_id, hidden=False)
    hidden = _fixture(
        random.Random(f"{spec.seed}:pipeline-hidden"),
        profile.profile_id,
        hidden=True,
    )
    correct_files = _correct_files(visible)
    bugs = _bugs(profile.profile_id)
    labels = [str(item["label"]) for item in bugs]
    profile_contract = {
        "d5_a": [
            "Normalize team names by trimming whitespace and case-folding.",
            "Exclude cancelled jobs before aggregation.",
        ],
        "d5_b": [
            "Resolve team aliases before aggregation.",
            "Deduplicate by job_id; the latest normalized updated_at record wins.",
        ],
        "d5_c": [
            "Resolve duplicate records before applying the team assignment effective at occurred_at.",
            "config/pipeline_config.json overrides config/default_pipeline.json.",
            "The final aggregation must consume artifacts/normalized_jobs.json.",
        ],
    }[profile.profile_id]
    return {
        "scenario_id": "team_hours_pipeline",
        "title": "Profiled Team Hours Pipeline",
        "required_output_path": "artifacts/summary.json",
        "debug_note": "The complete v2 contract and profile semantics are authoritative.",
        "hints": [
            "Run python run_pipeline.py and inspect both generated artifacts.",
            "The public check validates structure without revealing hidden rows.",
        ],
        "structure": {
            "repair_surface": "semantic_generalization",
            "bug_scope": "cross_file",
            "failure_mode": "operation_order_and_hidden_fixture",
            "smoke_test_quality": "partial_non_revealing",
            "d5_profile": profile.profile_id,
            "dependency_depth": profile.semantic_dependency_depth,
            "hidden_capability_count": 9,
            "distractor_count": 2,
        },
        "files": correct_files,
        "expected_output": _expected(visible),
        "normalized_output": _normalized(visible),
        "hidden_json_assets": {
            **{f"hidden_fixture/{path}": payload for path, payload in _fixture_json_files(hidden).items()},
            "hidden_expected_output.json": _expected(hidden),
        },
        "evaluator_config": {
            "hidden_fixture_dir": "hidden_fixture",
            "hidden_expected_path": "hidden_expected_output.json",
            "d5_profile": profile.profile_id,
        },
        "evaluator_entrypoint": "swg_generators.evaluators.pipeline_profile:ProfiledPipelineEvaluator",
        "profile_contract": profile_contract,
        "profile_managed_composition": True,
        "composition_spec": {
            "stages": [
                {
                    "stage_id": "canonicalize",
                    "required_inputs": [
                        "data/jobs.json",
                        "config/pipeline_config.json",
                    ],
                    "produced_artifacts": ["artifacts/normalized_jobs.json"],
                    "capability": "normalization",
                },
                {
                    "stage_id": "aggregate",
                    "required_inputs": ["artifacts/normalized_jobs.json"],
                    "produced_artifacts": ["artifacts/summary.json"],
                    "capability": "aggregation",
                },
            ],
            "dependencies": [["canonicalize", "aggregate"]],
            "stage_count": 2,
            "downstream_consumes_upstream_artifact": True,
        },
        "partial_solution_lattice_profile": {
            "no_fix_score": 0.15,
            "single_fix_max_score": 0.40,
            "pair_fix_max_score": 0.65,
            "all_but_one_max_score": 0.85,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "bugs": bugs,
        "defect_bundle": {
            "bundle_id": f"pipeline_{profile.profile_id}_semantic_chain",
            "defect_ids": labels,
            "dependency_edges": [[labels[index], labels[index + 1]] for index in range(len(labels) - 1)],
            "capability_groups": {label: [label] for label in labels},
            "required_files": sorted(
                {str(item["target_path"]) for item in bugs} | {"data/jobs.json", "config/pipeline_config.json"}
            ),
            "semantic_dependency_depth": profile.semantic_dependency_depth,
        },
    }


def _fixture(rng: random.Random, profile: str, *, hidden: bool) -> dict[str, object]:
    suffix = "H" if hidden else "V"
    base = round(rng.uniform(1.25, 4.25), 2)
    jobs: list[dict[str, object]] = [
        _job(f"{suffix}-1", " Platform ", "ready", base, "2026-02-01T09:00:00Z"),
        _job(f"{suffix}-2", "RESEARCH", "complete", 2.25, "2026-02-02T09:00:00+00:00"),
        _job(f"{suffix}-3", "ops", "cancelled", 9.0, "2026-02-03T09:00:00Z"),
    ]
    aliases: dict[str, str] = {}
    assignments: list[dict[str, object]] = []
    if profile in {"d5_b", "d5_c"}:
        aliases = {"PLAT": "platform", "RND": "research", "LEGACY-OPS": "ops"}
        duplicate_id = f"{suffix}-4"
        jobs.extend(
            [
                _job(duplicate_id, "PLAT", "ready", 1.0, "2026-02-04T10:30:00+00:00"),
                _job(duplicate_id, "plat", "ready", 3.5, "2026-02-04T06:00:00-05:00"),
                _job(f"{suffix}-5", "RND", "complete", 1.75, "2026-02-05T09:00:00Z"),
            ]
        )
    if profile == "d5_c":
        jobs.append(_job(f"{suffix}-6", "LEGACY-OPS", "ready", 4.0, "2026-02-15T12:00:00Z"))
        assignments = [
            {
                "job_id": f"{suffix}-6",
                "team": "platform",
                "effective_at": "2026-01-01T00:00:00Z",
            },
            {
                "job_id": f"{suffix}-6",
                "team": "research",
                "effective_at": "2026-03-01T00:00:00Z",
            },
        ]
        if hidden:
            assignments.insert(
                1,
                {
                    "job_id": f"{suffix}-6",
                    "team": "PLAT",
                    "effective_at": "2026-02-01T02:00:00+02:00",
                },
            )
    rng.shuffle(jobs)
    return {
        "jobs": jobs,
        "aliases": aliases,
        "assignments": assignments,
        "default_config": {
            "schema_version": "v1",
            "exclude_states": [],
            "output_path": "artifacts/legacy.json",
        },
        "local_config": {
            "schema_version": "v2",
            "input_path": "data/jobs.json",
            "aliases_path": "data/team_aliases.json",
            "assignments_path": "data/team_assignments.json",
            "exclude_states": ["cancelled"],
            "output_path": "artifacts/summary.json",
        },
    }


def _job(job_id: str, team: str, state: str, hours: float, updated_at: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "team": team,
        "state": state,
        "hours": hours,
        "occurred_at": updated_at,
        "updated_at": updated_at,
    }


def _parse(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _canonical_rows(fixture: dict[str, object]) -> list[dict[str, object]]:
    aliases = {str(k).strip().casefold(): str(v).strip().casefold() for k, v in dict(fixture["aliases"]).items()}
    latest: dict[str, dict[str, object]] = {}
    for raw in list(fixture["jobs"]):
        row = dict(raw)
        key = str(row["job_id"])
        if key not in latest or _parse(row["updated_at"]) > _parse(latest[key]["updated_at"]):
            latest[key] = row
    assignments: dict[str, list[dict[str, object]]] = {}
    for raw in list(fixture["assignments"]):
        row = dict(raw)
        assignments.setdefault(str(row["job_id"]), []).append(row)
    output = []
    for row in latest.values():
        candidates = [
            item
            for item in assignments.get(str(row["job_id"]), [])
            if _parse(item["effective_at"]) <= _parse(row["occurred_at"])
        ]
        team = str(candidates[-1]["team"] if candidates else row["team"]).strip().casefold()
        team = aliases.get(team, team)
        output.append(
            {
                "job_id": str(row["job_id"]),
                "team": team,
                "state": str(row["state"]).strip().casefold(),
                "hours": float(row["hours"]),
                "occurred_at": _parse(row["occurred_at"]).isoformat(),
                "updated_at": _parse(row["updated_at"]).isoformat(),
            }
        )
    return sorted(output, key=lambda row: str(row["job_id"]))


def _normalized(fixture: dict[str, object]) -> list[dict[str, object]]:
    return _canonical_rows(fixture)


def _expected(fixture: dict[str, object]) -> list[dict[str, object]]:
    excluded = {str(value).casefold() for value in fixture["local_config"]["exclude_states"]}
    summary: dict[str, dict[str, object]] = {}
    for row in _canonical_rows(fixture):
        if str(row["state"]) in excluded:
            continue
        team = str(row["team"])
        target = summary.setdefault(team, {"team": team, "job_count": 0, "total_hours": 0.0})
        target["job_count"] = int(target["job_count"]) + 1
        target["total_hours"] = round(float(target["total_hours"]) + float(row["hours"]), 1)
    return sorted(summary.values(), key=lambda row: str(row["team"]))


def _fixture_json_files(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "data/jobs.json": fixture["jobs"],
        "data/team_aliases.json": fixture["aliases"],
        "data/team_assignments.json": fixture["assignments"],
        "config/default_pipeline.json": fixture["default_config"],
        "config/pipeline_config.json": fixture["local_config"],
    }


def _correct_files(fixture: dict[str, object]) -> dict[str, str]:
    return {
        **{
            path: json.dumps(payload, indent=2, sort_keys=True) + "\n"
            for path, payload in _fixture_json_files(fixture).items()
        },
        "src/pipeline_app/__init__.py": "",
        "src/pipeline_app/io_utils.py": _io_utils(),
        "src/pipeline_app/steps.py": _steps(),
        "run_pipeline.py": _runner(),
        "public_check.py": _public_check(),
        "notes/legacy_pipeline.md": "Legacy v1 wrappers are retired and non-authoritative.\n",
        "config/pipeline_config.example.json": "{}\n",
    }


def _io_utils() -> str:
    return (
        dedent(
            """
        from __future__ import annotations
        import json
        from pathlib import Path

        def load_json(path: Path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        def write_json(path: Path, payload: object) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            temporary.replace(target)
        """
        ).strip()
        + "\n"
    )


def _steps() -> str:
    return (
        dedent(
            """
        from __future__ import annotations
        from datetime import datetime, timezone

        def normalize_text(value: object) -> str:
            return str(value).strip().casefold()

        def timestamp(value: object) -> datetime:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        def resolve_team(value: object, aliases: dict[str, str]) -> str:
            current = normalize_text(value)
            return aliases.get(current, current)

        def canonicalize(rows, aliases, assignments):
            latest = {}
            for row in rows:
                key = str(row["job_id"])
                if key not in latest or timestamp(row["updated_at"]) > timestamp(latest[key]["updated_at"]):
                    latest[key] = dict(row)
            history = {}
            for item in assignments:
                history.setdefault(str(item["job_id"]), []).append(item)
            normalized = []
            for row in latest.values():
                candidates = [item for item in history.get(str(row["job_id"]), []) if timestamp(item["effective_at"]) <= timestamp(row["occurred_at"])]
                source_team = candidates[-1]["team"] if candidates else row["team"]
                team = resolve_team(source_team, aliases)
                normalized.append({
                    "job_id": str(row["job_id"]),
                    "team": team,
                    "state": normalize_text(row["state"]),
                    "hours": float(row["hours"]),
                    "occurred_at": timestamp(row["occurred_at"]).isoformat(),
                    "updated_at": timestamp(row["updated_at"]).isoformat(),
                })
            return sorted(normalized, key=lambda row: row["job_id"])

        def build_summary(rows, exclude_states):
            excluded = {normalize_text(value) for value in exclude_states}
            summary = {}
            for row in rows:
                if row["state"] in excluded:
                    continue
                team = str(row["team"])
                target = summary.setdefault(team, {"team": team, "job_count": 0, "total_hours": 0.0})
                target["job_count"] += 1
                target["total_hours"] = round(target["total_hours"] + float(row["hours"]), 1)
            return sorted(summary.values(), key=lambda row: row["team"])
        """
        ).strip()
        + "\n"
    )


def _runner() -> str:
    return (
        dedent(
            """
        from __future__ import annotations
        import sys
        from pathlib import Path
        workspace = Path(__file__).resolve().parent
        sys.path.insert(0, str(workspace / "src"))
        from pipeline_app.io_utils import load_json, write_json
        from pipeline_app.steps import build_summary, canonicalize, normalize_text

        def main() -> None:
            defaults = load_json(workspace / "config/default_pipeline.json")
            local = load_json(workspace / "config/pipeline_config.json")
            config = {**defaults, **local}
            rows = load_json(workspace / config["input_path"])
            aliases = {normalize_text(k): normalize_text(v) for k, v in load_json(workspace / config["aliases_path"]).items()}
            assignments = load_json(workspace / config["assignments_path"])
            normalized = canonicalize(rows, aliases, assignments)
            normalized_path = workspace / "artifacts/normalized_jobs.json"
            write_json(normalized_path, normalized)
            summary = build_summary(load_json(normalized_path), config["exclude_states"])
            write_json(workspace / config["output_path"], summary)

        if __name__ == "__main__":
            main()
        """
        ).strip()
        + "\n"
    )


def _public_check() -> str:
    return (
        dedent(
            """
        import json
        from pathlib import Path
        target = Path("artifacts/summary.json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert all(set(row) == {"team", "job_count", "total_hours"} for row in payload)
        assert [row["team"] for row in payload] == sorted(row["team"] for row in payload)
        print("public structure check passed")
        """
        ).strip()
        + "\n"
    )


def _replace(old: str, new: str, *, label: str, target_path: str):
    def apply(content: str) -> str:
        updated = content.replace(old, new, 1)
        if updated == content:
            raise ValueError(f"pipeline profile defect {label!r} did not modify {target_path}")
        return updated

    return apply


def _bugs(profile: str) -> list[dict[str, object]]:
    candidates = {
        "normalization_bypassed": {
            "label": "normalization_bypassed",
            "target_path": "src/pipeline_app/steps.py",
            "apply": _replace(
                "team = resolve_team(source_team, aliases)",
                "team = str(source_team)",
                label="normalization_bypassed",
                target_path="src/pipeline_app/steps.py",
            ),
        },
        "cancelled_filter_disabled": {
            "label": "cancelled_filter_disabled",
            "target_path": "config/pipeline_config.json",
            "apply": _replace(
                '"cancelled"',
                '"archived"',
                label="cancelled_filter_disabled",
                target_path="config/pipeline_config.json",
            ),
        },
        "alias_resolution_bypassed": {
            "label": "alias_resolution_bypassed",
            "target_path": "run_pipeline.py",
            "apply": _replace(
                'aliases = {normalize_text(k): normalize_text(v) for k, v in load_json(workspace / config["aliases_path"]).items()}',
                "aliases = {}",
                label="alias_resolution_bypassed",
                target_path="run_pipeline.py",
            ),
        },
        "deduplication_disabled": {
            "label": "deduplication_disabled",
            "target_path": "src/pipeline_app/steps.py",
            "apply": _replace(
                "for row in latest.values():",
                "for row in rows:",
                label="deduplication_disabled",
                target_path="src/pipeline_app/steps.py",
            ),
        },
        "lexical_timestamp_comparison": {
            "label": "lexical_timestamp_comparison",
            "target_path": "src/pipeline_app/steps.py",
            "apply": _replace(
                'timestamp(row["updated_at"]) > timestamp(latest[key]["updated_at"])',
                'str(row["updated_at"]) > str(latest[key]["updated_at"])',
                label="lexical_timestamp_comparison",
                target_path="src/pipeline_app/steps.py",
            ),
        },
        "temporal_assignment_ignored": {
            "label": "temporal_assignment_ignored",
            "target_path": "src/pipeline_app/steps.py",
            "apply": _replace(
                'if timestamp(item["effective_at"]) <= timestamp(row["occurred_at"])',
                "if True",
                label="temporal_assignment_ignored",
                target_path="src/pipeline_app/steps.py",
            ),
        },
        "config_precedence_reversed": {
            "label": "config_precedence_reversed",
            "target_path": "run_pipeline.py",
            "apply": _replace(
                "config = {**defaults, **local}",
                "config = {**local, **defaults}",
                label="config_precedence_reversed",
                target_path="run_pipeline.py",
            ),
        },
        "intermediate_artifact_bypassed": {
            "label": "intermediate_artifact_bypassed",
            "target_path": "run_pipeline.py",
            "apply": _replace(
                "write_json(normalized_path, normalized)",
                "write_json(normalized_path, rows)",
                label="intermediate_artifact_bypassed",
                target_path="run_pipeline.py",
            ),
        },
    }
    selected = {
        "d5_a": ["normalization_bypassed", "cancelled_filter_disabled"],
        "d5_b": [
            "alias_resolution_bypassed",
            "deduplication_disabled",
            "lexical_timestamp_comparison",
        ],
        "d5_c": [
            "config_precedence_reversed",
            "intermediate_artifact_bypassed",
            "alias_resolution_bypassed",
            "deduplication_disabled",
            "lexical_timestamp_comparison",
            "temporal_assignment_ignored",
        ],
    }[profile]
    return [candidates[label] for label in selected]
