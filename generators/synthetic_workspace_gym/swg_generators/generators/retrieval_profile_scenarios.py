from __future__ import annotations

import random
from copy import deepcopy
from textwrap import dedent

from swg_generators.generators.d5_profiles import (
    D5Profile,
    select_weighted_d5_profile,
)
from swg_generators.generators.retrieval_workspace_scenarios import (
    add_distractor_documents,
    base_profile,
    build_client_adapter_hidden_runner,
    build_client_adapter_sync_scenario,
    client_adapter_variants,
    document_roots,
    render_json,
    select_fixture_variant,
)
from swg_generators.schemas import EnvironmentSpec

RETRIEVAL_PROFILE_SCENARIOS = {
    "d5_a": "client_adapter_sync",
    "d5_b": "client_adapter_policy_sync",
    "d5_c": "versioned_client_migration",
}

CAPABILITY_WEIGHTS = {
    "authority_resolution": 0.10,
    "schema_mapping": 0.15,
    "quantity_parsing": 0.10,
    "pagination": 0.10,
    "missing_value_policy": 0.10,
    "regional_override": 0.10,
    "deduplication": 0.10,
    "timestamp_resolution": 0.10,
    "output_contract": 0.05,
    "hidden_generalization": 0.10,
}


def retrieval_profile(spec: EnvironmentSpec) -> D5Profile:
    profile = select_weighted_d5_profile(spec.difficulty, spec.seed)
    if profile is None:
        raise ValueError("profiled retrieval scenarios require difficulty 5")
    return profile


def retrieval_scenario_id(spec: EnvironmentSpec) -> str:
    return RETRIEVAL_PROFILE_SCENARIOS[retrieval_profile(spec).profile_id]


def build_profiled_retrieval_scenario(
    rng: random.Random,
    spec: EnvironmentSpec,
    *,
    scenario_id: str | None = None,
) -> dict[str, object]:
    selected = scenario_id or retrieval_scenario_id(spec)
    if selected == "client_adapter_sync":
        return build_recoverable_adapter_scenario(rng, spec)
    if selected == "client_adapter_policy_sync":
        return _build_policy_scenario(rng, spec)
    if selected == "versioned_client_migration":
        return _build_versioned_scenario(rng, spec)
    raise ValueError(f"unknown profiled retrieval scenario: {selected}")


def _recoverable_run_example() -> str:
    return (
        dedent(
            """
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        workspace = Path(__file__).resolve().parent
        sys.path.insert(0, str(workspace / "src"))
        from client_adapter import build_summary

        payload = json.loads(
            (workspace / "samples" / "response.json").read_text(encoding="utf-8")
        )
        summary = build_summary(payload)
        records = list(payload["records"])
        expected_keys = {
            "request_id", "next_cursor", "record_count", "total_quantity", "warehouses"
        }
        assert set(summary) == expected_keys
        assert summary["request_id"] == str(payload["request_id"])
        assert summary["next_cursor"] == payload.get("next_cursor")
        assert isinstance(summary["record_count"], int)
        assert isinstance(summary["total_quantity"], int)
        assert isinstance(summary["warehouses"], list)
        print(json.dumps(summary, indent=2, sort_keys=True))
        """
        ).strip()
        + "\n"
    )


def _correct_recoverable_adapter() -> str:
    return (
        dedent(
            """
        from __future__ import annotations


        def build_summary(response: dict[str, object]) -> dict[str, object]:
            records: list[dict[str, object]] = []
            for candidate in response.get("records", []):
                if not isinstance(candidate, dict):
                    continue
                try:
                    quantity = int(str(candidate["quantity"]).strip())
                except (KeyError, TypeError, ValueError):
                    continue
                records.append({**candidate, "quantity": quantity})
            return {
                "request_id": str(response["request_id"]),
                "next_cursor": response.get("next_cursor"),
                "record_count": len(records),
                "total_quantity": sum(int(record["quantity"]) for record in records),
                "warehouses": sorted(
                    {str(record.get("warehouse", "unknown")) for record in records}
                ),
            }
        """
        ).strip()
        + "\n"
    )


def build_recoverable_adapter_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    """Keep the original adapter as the recoverable D5-A profile.

    The semantic surface stays compatible, but the visible evidence is split by
    interface/policy role instead of presenting a patch recipe.
    """

    scenario = deepcopy(build_client_adapter_sync_scenario(rng, spec))
    files = dict(scenario["files"])
    correct_adapter = _correct_recoverable_adapter()
    correct_files = dict(scenario["correct_files"])
    correct_files["src/client_adapter.py"] = correct_adapter
    reference_solution_files = dict(scenario["reference_solution_files"])
    reference_solution_files["src/client_adapter.py"] = correct_adapter
    for path in (
        "docs/current_api_summary.md",
        "docs/validation_guide.md",
        "docs/version_applicability.md",
        "docs/field_mapping.md",
        "docs/warehouse_override.md",
    ):
        files.pop(path, None)
    files["docs/api_reference.md"] = (
        dedent(
            """
        Evidence bundle: `client-api-2026-03`

        # Client API v3 response

        The current response exposes a `records` collection. Each record has a
        `sku`, a numeric `quantity`, and may include `warehouse`.
        """
        ).strip()
        + "\n"
    )
    files["docs/pagination_contract.md"] = (
        dedent(
            """
        # Pagination contract

        Continuation is represented by `next_cursor`. A null value denotes the
        final page.
        """
        ).strip()
        + "\n"
    )
    files["notes/warehouse_policy.md"] = (
        dedent(
            """
        # Warehouse grouping policy

        Records without an assigned warehouse belong to the `unknown` group.
        Summary warehouse groups are unique and deterministic.
        """
        ).strip()
        + "\n"
    )
    files["notes/record_quality.md"] = (
        dedent(
            """
        # Record quality contract

        A current record is a mapping with a quantity that can be safely coerced
        from an integer or a stripped integer string. Ignore non-mapping records,
        missing quantities, and non-numeric quantities before computing count,
        total, or warehouse groups. One malformed record must not reject its page.
        """
        ).strip()
        + "\n"
    )
    files["run_example.py"] = _recoverable_run_example()
    files["changelog/client_migration.md"] = (
        dedent(
            """
        # March client cutover

        The v3 response and pagination contracts became authoritative in March.
        Pre-v3 collection, quantity, and continuation names are incompatible.
        """
        ).strip()
        + "\n"
    )
    count_contract_bug = _bug(
        "output_count_contract",
        "src/client_adapter.py",
        '"record_count": len(records)',
        '"item_count": len(records)',
    )
    malformed_record_bug = _bug(
        "malformed_record_handling",
        "src/client_adapter.py",
        """try:
            quantity = int(str(candidate["quantity"]).strip())
        except (KeyError, TypeError, ValueError):
            continue""",
        'quantity = int(candidate["quantity"])',
    )
    scenario["bugs"] = [
        *list(scenario["bugs"]),
        count_contract_bug,
        malformed_record_bug,
    ]
    files["src/client_adapter.py"] = files["src/client_adapter.py"].replace(
        '"record_count": len(items)', '"item_count": len(items)', 1
    )
    scenario["correct_files"] = correct_files
    scenario["reference_solution_files"] = reference_solution_files
    variant = select_fixture_variant(spec.seed, client_adapter_variants())
    hidden_assets = dict(scenario["hidden_text_assets"])
    hidden_assets["run_hidden_tests.py"] = build_client_adapter_hidden_runner(variant, include_malformed_records=True)
    scenario["hidden_text_assets"] = hidden_assets
    defect_bundle = dict(scenario["defect_bundle"])
    defect_bundle["defect_ids"] = [
        *list(defect_bundle["defect_ids"]),
        "output_count_contract",
        "malformed_record_handling",
    ]
    defect_bundle["capability_groups"] = {
        **dict(defect_bundle["capability_groups"]),
        "output_contract": ["output_count_contract"],
        "record_quality": ["malformed_record_handling"],
    }
    scenario.update(
        {
            "files": files,
            "title": "Client Adapter Sync (Recoverable)",
            "description": (
                "Repair the single adapter by reconciling the active API, pagination, and warehouse policy documents."
            ),
            "hints": [
                "The active interface is distributed across API, pagination, and policy documents.",
                "The January client note is historical rather than authoritative.",
            ],
            "structure": {
                **dict(scenario["structure"]),
                "d5_profile": "d5_a",
                "semantic_dependency_depth": 2,
                "capability_count": 14,
                "solution_artifact_count": 2,
            },
            "document_roots": document_roots(files),
            "defect_bundle": defect_bundle,
            "unmodified_reward_limit": 0.15,
        }
    )
    return scenario


def _correct_config() -> dict[str, object]:
    return {
        "api_version": "v3",
        "collection_field": "records",
        "quantity_field": "quantity",
        "cursor_field": "next_cursor",
        "region": "eu-west",
        "missing_warehouse_policy": "unknown",
        "warehouse_aliases": {
            "eu-west": {"ams-old": "eu-central", "dub-legacy": "eu-west"},
            "us-east": {"iad-old": "us-east"},
        },
    }


def _stale_config() -> dict[str, object]:
    return {
        "api_version": "v2",
        "collection_field": "items",
        "quantity_field": "count",
        "cursor_field": "cursor",
        "region": "global",
        "missing_warehouse_policy": "error",
        "warehouse_aliases": {},
    }


def _correct_parser(config_name: str) -> str:
    return (
        dedent(
            f'''
        from __future__ import annotations

        import json
        from datetime import datetime
        from pathlib import Path


        CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "{config_name}"


        def _load_config(config: dict[str, object] | None) -> dict[str, object]:
            if config is not None:
                return config
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


        def _timestamp(value: object) -> datetime:
            text = str(value or "1970-01-01T00:00:00Z").replace("Z", "+00:00")
            return datetime.fromisoformat(text)


        def normalize_response(
            response: dict[str, object],
            *,
            config: dict[str, object] | None = None,
        ) -> dict[str, object]:
            config = _load_config(config)
            raw_records = list(response.get(config["collection_field"], []))
            region = str(config["region"])
            aliases = dict(dict(config.get("warehouse_aliases", {{}})).get(region, {{}}))
            latest: dict[str, tuple[datetime, dict[str, object]]] = {{}}
            for raw in raw_records:
                if not isinstance(raw, dict) or not raw.get("sku"):
                    continue
                raw_quantity = raw.get(config["quantity_field"])
                try:
                    quantity = int(str(raw_quantity).strip())
                except (TypeError, ValueError):
                    continue
                warehouse = raw.get("warehouse")
                if warehouse in (None, ""):
                    warehouse = config["missing_warehouse_policy"]
                warehouse = aliases.get(str(warehouse), str(warehouse))
                record = {{
                    "sku": str(raw["sku"]).strip().upper(),
                    "quantity": quantity,
                    "warehouse": warehouse,
                    "updated_at": str(raw.get("updated_at", "1970-01-01T00:00:00Z")),
                }}
                stamp = _timestamp(record["updated_at"])
                current = latest.get(record["sku"])
                if current is None or stamp > current[0]:
                    latest[record["sku"]] = (stamp, record)
            records = [item[1] for item in sorted(latest.values(), key=lambda item: item[1]["sku"])]
            return {{
                "request_id": str(response.get("request_id", "")),
                "next_cursor": response.get(config["cursor_field"]),
                "records": records,
            }}
        '''
        ).strip()
        + "\n"
    )


def _correct_summary(parser_module: str) -> str:
    return (
        dedent(
            f"""
        from __future__ import annotations

        from {parser_module} import normalize_response


        def build_summary(
            response: dict[str, object],
            *,
            config: dict[str, object] | None = None,
        ) -> dict[str, object]:
            canonical = normalize_response(response, config=config)
            records = list(canonical["records"])
            return {{
                "request_id": canonical["request_id"],
                "next_cursor": canonical["next_cursor"],
                "record_count": len(records),
                "total_quantity": sum(int(record["quantity"]) for record in records),
                "warehouses": sorted({{str(record["warehouse"]) for record in records}}),
            }}
        """
        ).strip()
        + "\n"
    )


def _correct_record_policy() -> str:
    return (
        dedent(
            """
        from __future__ import annotations

        from schema_policy import quantity_from_record
        from temporal_policy import should_replace


        def canonicalize_records(
            raw_records: list[object],
            config: dict[str, object],
        ) -> list[dict[str, object]]:
            region = str(config["region"])
            aliases = dict(dict(config.get("warehouse_aliases", {})).get(region, {}))
            latest: dict[str, dict[str, object]] = {}
            for raw in raw_records:
                if not isinstance(raw, dict) or not raw.get("sku"):
                    continue
                raw_quantity = quantity_from_record(raw, config)
                try:
                    quantity = int(str(raw_quantity).strip())
                except (TypeError, ValueError):
                    continue
                warehouse = raw.get("warehouse")
                if warehouse in (None, ""):
                    warehouse = config["missing_warehouse_policy"]
                warehouse = aliases.get(str(warehouse), str(warehouse))
                record = {
                    "sku": str(raw["sku"]).strip().upper(),
                    "quantity": quantity,
                    "warehouse": warehouse,
                    "updated_at": str(raw.get("updated_at", "1970-01-01T00:00:00Z")),
                }
                current = latest.get(record["sku"])
                current_stamp = None if current is None else current["updated_at"]
                if should_replace(current_stamp, record["updated_at"]):
                    latest[record["sku"]] = record
            return sorted(latest.values(), key=lambda item: item["sku"])
        """
        ).strip()
        + "\n"
    )


def _correct_schema_policy() -> str:
    return (
        dedent(
            """
        from __future__ import annotations


        def records_from_response(
            response: dict[str, object], config: dict[str, object]
        ) -> list[object]:
            return list(response.get(config["collection_field"], []))


        def quantity_from_record(
            record: dict[str, object], config: dict[str, object]
        ) -> object:
            return record.get(config["quantity_field"])
        """
        ).strip()
        + "\n"
    )


def _correct_temporal_policy() -> str:
    return (
        dedent(
            """
        from __future__ import annotations

        from datetime import datetime


        def _timestamp(value: object) -> datetime:
            text = str(value or "1970-01-01T00:00:00Z").replace("Z", "+00:00")
            return datetime.fromisoformat(text)


        def should_replace(current: object | None, candidate: object) -> bool:
            if current is None:
                return True
            return _timestamp(candidate) > _timestamp(current)
        """
        ).strip()
        + "\n"
    )


def _correct_versioned_parser(config_name: str) -> str:
    return (
        dedent(
            f"""
        from __future__ import annotations

        import json
        from pathlib import Path

        from record_policy import canonicalize_records
        from schema_policy import records_from_response


        CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "{config_name}"


        def _load_config(config: dict[str, object] | None) -> dict[str, object]:
            if config is not None:
                return config
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


        def normalize_response(
            response: dict[str, object],
            *,
            config: dict[str, object] | None = None,
        ) -> dict[str, object]:
            config = _load_config(config)
            raw_records = records_from_response(response, config)
            records = canonicalize_records(raw_records, config)
            return {{
                "request_id": str(response.get("request_id", "")),
                "next_cursor": response.get(config["cursor_field"]),
                "records": records,
            }}
        """
        ).strip()
        + "\n"
    )


def _bug(label: str, target_path: str, old: str, new: str) -> dict[str, object]:
    def apply(content: str) -> str:
        updated = content.replace(old, new, 1)
        if updated == content:
            raise ValueError(f"retrieval defect {label!r} did not modify {target_path}")
        return updated

    return {"label": label, "target_path": target_path, "apply": apply}


def _config_bug(target_path: str) -> dict[str, object]:
    correct = render_json(_correct_config())
    stale = render_json(_stale_config())
    return _bug("authority_config", target_path, correct, stale)


def _profile_bugs(*, parser_path: str, summary_path: str, config_path: str) -> list[dict[str, object]]:
    return [
        _config_bug(config_path),
        _bug(
            "quantity_parsing",
            parser_path,
            'raw.get(config["quantity_field"])',
            'raw.get("count")',
        ),
        _bug(
            "missing_value_policy",
            parser_path,
            'warehouse = config["missing_warehouse_policy"]',
            'warehouse = "error"',
        ),
        _bug(
            "regional_override",
            parser_path,
            'aliases = dict(dict(config.get("warehouse_aliases", {})).get(region, {}))',
            "aliases = {}",
        ),
        _bug(
            "deduplication",
            parser_path,
            'latest[record["sku"]] = (stamp, record)',
            "latest[f\"{record['sku']}:{len(latest)}\"] = (stamp, record)",
        ),
        _bug(
            "timestamp_resolution",
            parser_path,
            "if current is None or stamp > current[0]:",
            "if current is None:",
        ),
        _bug(
            "pagination",
            summary_path,
            '"next_cursor": canonical["next_cursor"]',
            '"next_cursor": response.get("cursor")',
        ),
        _bug(
            "output_contract",
            summary_path,
            '"record_count": len(records)',
            '"item_count": len(records)',
        ),
    ]


def _versioned_profile_bugs(
    *,
    parser_path: str,
    policy_path: str,
    schema_path: str,
    temporal_path: str,
    summary_path: str,
    config_path: str,
) -> list[dict[str, object]]:
    return [
        _config_bug(config_path),
        _bug(
            "schema_mapping",
            parser_path,
            "records_from_response(response, config)",
            'list(response.get("items", []))',
        ),
        _bug(
            "quantity_parsing",
            schema_path,
            'record.get(config["quantity_field"])',
            'record.get("count")',
        ),
        _bug(
            "missing_value_policy",
            policy_path,
            'warehouse = config["missing_warehouse_policy"]',
            'warehouse = "error"',
        ),
        _bug(
            "regional_override",
            policy_path,
            'aliases = dict(dict(config.get("warehouse_aliases", {})).get(region, {}))',
            "aliases = {}",
        ),
        _bug(
            "deduplication",
            policy_path,
            'latest[record["sku"]] = record',
            "latest[f\"{record['sku']}:{len(latest)}\"] = record",
        ),
        _bug(
            "timestamp_resolution",
            temporal_path,
            "return _timestamp(candidate) > _timestamp(current)",
            "return _timestamp(candidate) < _timestamp(current)",
        ),
        _bug(
            "pagination",
            summary_path,
            '"next_cursor": canonical["next_cursor"]',
            '"next_cursor": response.get("cursor")',
        ),
        _bug(
            "output_contract",
            summary_path,
            '"record_count": len(records)',
            '"item_count": len(records)',
        ),
    ]


def _apply_all_bugs(correct_files: dict[str, str], bugs: list[dict[str, object]]) -> dict[str, str]:
    files = dict(correct_files)
    for bug in bugs:
        path = str(bug["target_path"])
        files[path] = bug["apply"](files[path])
    return files


def _hidden_runner(
    *,
    parser_module: str,
    summary_module: str,
    config_path: str,
) -> str:
    return (
        dedent(
            f'''
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path


        CAPABILITY_WEIGHTS = {CAPABILITY_WEIGHTS!r}
        EXPECTED_KEYS = {{"request_id", "next_cursor", "record_count", "total_quantity", "warehouses"}}
        EXPECTED_CONFIG = {_correct_config()!r}


        def main() -> None:
            workspace = Path(sys.argv[1]).resolve()
            sys.path.insert(0, str(workspace / "src"))
            try:
                from {parser_module} import normalize_response
                from {summary_module} import build_summary
            except Exception as exc:
                print(json.dumps({{"import_error": repr(exc)}}))
                raise

            def check(callback) -> float:
                try:
                    return float(bool(callback()))
                except Exception:
                    return 0.0

            config_file = workspace / "{config_path}"
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                config = {{}}

            shared = {{
                "request_id": "shared",
                "next_cursor": "cursor-v3",
                "cursor": "cursor-v2",
                "records": [
                    {{"sku": "a-1", "quantity": "2", "warehouse": "ams-old", "updated_at": "2026-04-01T09:00:00Z"}},
                    {{"sku": "A-1", "quantity": "7", "warehouse": "dub-legacy", "updated_at": "2026-04-01T10:00:00+00:00"}},
                    {{"sku": "b-2", "quantity": 3, "updated_at": "2026-04-01T11:00:00Z"}},
                ],
                "items": [{{"sku": "legacy", "count": 99, "warehouse": "legacy"}}],
            }}
            capabilities = {{
                "authority_resolution": check(lambda: config.get("api_version") == "v3" and config.get("region") == "eu-west"),
                "schema_mapping": check(lambda: [r["sku"] for r in normalize_response({{"request_id": "s", "records": [{{"sku": "x", "quantity": 1, "count": 9}}], "items": [{{"sku": "bad", "count": 5}}], "next_cursor": None}}, config=EXPECTED_CONFIG)["records"]] == ["X"]),
                "quantity_parsing": check(lambda: build_summary({{"request_id": "q", "records": [{{"sku": "x", "quantity": "2"}}, {{"sku": "y", "quantity": 5}}], "next_cursor": None}}, config=EXPECTED_CONFIG)["total_quantity"] == 7),
                "pagination": check(lambda: build_summary(shared, config=EXPECTED_CONFIG)["next_cursor"] == "cursor-v3"),
                "missing_value_policy": check(lambda: build_summary({{"request_id": "m", "records": [{{"sku": "x", "quantity": 1}}], "next_cursor": None}}, config=EXPECTED_CONFIG)["warehouses"] == ["unknown"]),
                "regional_override": check(lambda: build_summary({{"request_id": "r", "records": [{{"sku": "x", "quantity": 1, "warehouse": "ams-old"}}], "next_cursor": None}}, config=EXPECTED_CONFIG)["warehouses"] == ["eu-central"]),
                "deduplication": check(lambda: build_summary(shared, config=EXPECTED_CONFIG)["record_count"] == 2),
                "timestamp_resolution": check(lambda: build_summary(shared, config=EXPECTED_CONFIG)["total_quantity"] == 10),
                "output_contract": check(lambda: set(build_summary(shared, config=EXPECTED_CONFIG)) == EXPECTED_KEYS),
            }}
            alternate_config = {{
                **config,
                "api_version": "v3",
                "collection_field": "entries",
                "quantity_field": "units",
                "cursor_field": "continuation",
                "region": "us-east",
                "missing_warehouse_policy": "unassigned",
                "warehouse_aliases": {{"us-east": {{"iad-old": "us-east"}}}},
            }}
            alternate = {{
                "request_id": "alternate",
                "continuation": None,
                "entries": [
                    {{"sku": "z", "units": "4", "warehouse": "iad-old", "updated_at": "2026-05-01T00:00:00Z"}},
                    {{"sku": "bad", "units": "not-numeric"}},
                    {{"units": 8}},
                ],
            }}
            capabilities["hidden_generalization"] = check(
                lambda: build_summary(alternate, config=alternate_config) == {{
                    "request_id": "alternate",
                    "next_cursor": None,
                    "record_count": 1,
                    "total_quantity": 4,
                    "warehouses": ["us-east"],
                }}
            )
            passed = sum(capabilities.values())
            success = all(value == 1.0 for value in capabilities.values())
            print(json.dumps({{
                "success": success,
                "subscores": {{
                    **{{f"capability_{{name}}": value for name, value in capabilities.items()}},
                    "tests_passed": passed,
                    "tests_total": len(capabilities),
                }},
                "capability_weights": CAPABILITY_WEIGHTS,
                "failure_labels": [] if success else ["hidden_capabilities_failed"],
                "diagnostics": {{"failed_capabilities": [name for name, value in capabilities.items() if value < 1.0]}},
            }}, sort_keys=True))


        if __name__ == "__main__":
            main()
        '''
        ).strip()
        + "\n"
    )


def _run_example(
    summary_module: str,
    *,
    expected_warehouses: list[str] | None = None,
    check_config_override: bool = False,
) -> str:
    return (
        dedent(
            f"""
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        workspace = Path(__file__).resolve().parent
        sys.path.insert(0, str(workspace / "src"))
        from {summary_module} import build_summary

        payload = json.loads((workspace / "samples" / "response.json").read_text(encoding="utf-8"))
        summary = build_summary(payload)
        expected_keys = {{
            "request_id",
            "next_cursor",
            "record_count",
            "total_quantity",
            "warehouses",
        }}
        assert set(summary) == expected_keys, (
            f"summary keys must be {{sorted(expected_keys)}}, got {{sorted(summary)}}"
        )
        assert summary["request_id"] == "visible-1"
        assert summary["next_cursor"] == "visible-next"
        assert summary["record_count"] == 2
        assert summary["total_quantity"] == 5
        warehouses = summary["warehouses"]
        assert isinstance(warehouses, list)
        assert warehouses == sorted(set(warehouses))
        assert len(warehouses) == 2
        expected_warehouses = {expected_warehouses!r}
        if expected_warehouses is not None:
            assert warehouses == expected_warehouses
        if {check_config_override!r}:
            override = {{
                "api_version": "v3",
                "collection_field": "entries",
                "quantity_field": "units",
                "cursor_field": "continuation",
                "region": "test",
                "missing_warehouse_policy": "unassigned",
                "warehouse_aliases": {{"test": {{}}}},
            }}
            alternate = build_summary(
                {{
                    "request_id": "visible-override",
                    "continuation": None,
                    "entries": [{{"sku": "X-1", "units": "4"}}],
                }},
                config=override,
            )
            assert alternate == {{
                "request_id": "visible-override",
                "next_cursor": None,
                "record_count": 1,
                "total_quantity": 4,
                "warehouses": ["unassigned"],
            }}
        print(json.dumps(summary, indent=2, sort_keys=True))
        """
        ).strip()
        + "\n"
    )


def _sample_payload() -> dict[str, object]:
    return {
        "request_id": "visible-1",
        "next_cursor": "visible-next",
        "records": [
            {
                "sku": "A-1",
                "quantity": "2",
                "warehouse": "ams-old",
                "updated_at": "2026-04-01T10:00:00Z",
            },
            {
                "sku": "B-2",
                "quantity": 3,
                "updated_at": "2026-04-01T11:00:00Z",
            },
        ],
    }


def _policy_sample_payload() -> dict[str, object]:
    return {
        "request_id": "visible-1",
        "next_cursor": "visible-next",
        "records": [
            {
                "sku": "a-1",
                "quantity": "2",
                "warehouse": "ams-old",
                "updated_at": "2026-04-01T09:00:00Z",
            },
            {
                "sku": "A-1",
                "quantity": 2,
                "warehouse": "dub-legacy",
                "updated_at": "2026-04-01T11:00:00+00:00",
            },
            {"sku": "B-2", "quantity": 3, "updated_at": "2026-04-01T10:00:00Z"},
            {"sku": "bad", "quantity": "not-numeric"},
        ],
    }


def _build_policy_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    profile = retrieval_profile(spec)
    parser_path = "src/client_parser.py"
    summary_path = "src/client_summary.py"
    config_path = "config/client_runtime.json"
    correct_files = {
        config_path: render_json(_correct_config()),
        parser_path: _correct_parser("client_runtime.json"),
        summary_path: _correct_summary("client_parser"),
    }
    bugs = _profile_bugs(
        parser_path=parser_path,
        summary_path=summary_path,
        config_path=config_path,
    )
    files = _apply_all_bugs(correct_files, bugs)
    files.update(
        {
            "samples/response.json": render_json(_policy_sample_payload()),
            "run_example.py": _run_example(
                "client_summary",
                expected_warehouses=["eu-west", "unknown"],
                check_config_override=True,
            ),
            "release/current_manifest.json": render_json(
                {
                    "active_api": "v3",
                    "region": "eu-west",
                    "cutover": "2026-03-01",
                    "policy_bundle": "warehouse-policy-2026-03",
                }
            ),
            "docs/api_response_v3.md": dedent(
                """
                # API response v3

                Responses contain `request_id`, `records`, and `next_cursor`.
                Each record has `sku`, `quantity`, optional `warehouse`, and an
                ISO-8601 `updated_at`. Quantity may be an integer or numeric string.
                """
            ).strip()
            + "\n",
            "docs/pagination_contract.md": "# Pagination\n\n`next_cursor` is null on the final page.\n",
            "changelog/client_migration.md": dedent(
                """
                # Client migration

                The March cutover activates API v3. The release manifest is the
                authority for the active API, region, and policy bundle. Runtime
                configuration, parser, and summary code must agree on that interface.
                """
            ).strip()
            + "\n",
            "docs/authority_precedence.md": dedent(
                """
                # Client evidence precedence

                Resolve conflicts in this order: the current release manifest,
                the policy bundle named by that manifest, current API and output
                contracts, then archived rollout notes. Archived notes describe
                historical behavior and never override the active manifest.
                """
            ).strip()
            + "\n",
            "policies/warehouse-policy-2026-03.md": dedent(
                """
                # Warehouse policy bundle: warehouse-policy-2026-03

                The active missing-warehouse value is `unknown`.
                For `eu-west`, normalize `ams-old` to `eu-central` and
                `dub-legacy` to `eu-west`. Runtime configuration stores aliases
                by region, and parser behavior must derive from that configuration.
                """
            ).strip()
            + "\n",
            "notes/warehouse_policy.md": dedent(
                """
                # Warehouse policy implementation note

                Missing assignments use the active configured policy value.
                Region-specific aliases in runtime configuration are applied before
                summary grouping. Consult the manifest-selected policy bundle for
                the current values.
                """
            ).strip()
            + "\n",
            "notes/record_identity.md": dedent(
                """
                # Record identity

                SKU identity is `str(sku).strip().upper()`. Maintain one record per
                normalized key. Parse `updated_at` as a timezone-aware instant; store
                the first valid candidate, replace it only when a later instant arrives,
                and retain the current record for equal or older instants. Perform this
                resolution before aggregation. Malformed records are ignored and output
                is deterministic. This algorithm is normative for the active bundle.
                """
            ).strip()
            + "\n",
            "notes/legacy_rollout.md": dedent(
                """
                # Legacy rollout (archived)

                The pre-cutover client read `items`, `count`, and `cursor`, used the
                first duplicate, and treated warehouse omission as an error. This
                behavior is retained for historical diagnosis only and is superseded
                by the current release manifest and its selected policy bundle.
                """
            ).strip()
            + "\n",
            "docs/summary_contract.md": dedent(
                """
                # Summary contract

                Emit request_id, next_cursor, record_count, total_quantity, and
                unique warehouses in ascending order.
                """
            ).strip()
            + "\n",
        }
    )
    add_distractor_documents(files, rng, 4)
    return _hard_scenario(
        scenario_id="client_adapter_policy_sync",
        title="Client Adapter Policy Sync",
        description="Synthesize distributed API and policy evidence across parser, summary, and runtime configuration.",
        profile=profile,
        files=files,
        correct_files=correct_files,
        bugs=bugs,
        parser_module="client_parser",
        summary_module="client_summary",
        parser_path=parser_path,
        summary_path=summary_path,
        config_path=config_path,
    )


def _build_versioned_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    profile = retrieval_profile(spec)
    parser_path = "src/adapter.py"
    policy_path = "src/record_policy.py"
    schema_path = "src/schema_policy.py"
    temporal_path = "src/temporal_policy.py"
    summary_path = "src/serializer.py"
    config_path = "config/client.json"
    correct_files = {
        config_path: render_json(_correct_config()),
        schema_path: _correct_schema_policy(),
        temporal_path: _correct_temporal_policy(),
        policy_path: _correct_record_policy(),
        parser_path: _correct_versioned_parser("client.json"),
        summary_path: _correct_summary("adapter"),
    }
    bugs = _versioned_profile_bugs(
        parser_path=parser_path,
        policy_path=policy_path,
        schema_path=schema_path,
        temporal_path=temporal_path,
        summary_path=summary_path,
        config_path=config_path,
    )
    files = _apply_all_bugs(correct_files, bugs)
    files.update(
        {
            "samples/response.json": render_json(_sample_payload()),
            "run_example.py": _run_example("serializer"),
            "release/current_manifest.json": render_json(
                {
                    "active_api": "v3",
                    "region": "eu-west",
                    "cutover": "2026-03-01",
                    "policy_bundle": "warehouse-policy-2026-03",
                }
            ),
            "docs/api_v2.md": dedent(
                """
                # API v2 (supported only before cutover)

                v2 exposes items, count, and cursor. Missing warehouse is invalid.
                """
            ).strip()
            + "\n",
            "docs/api_v3.md": dedent(
                """
                # API v3

                v3 exposes records and next_cursor. Records contain SKU, quantity,
                optional warehouse, and updated_at. Numeric strings are accepted.
                """
            ).strip()
            + "\n",
            "changelog/v3_cutover.md": dedent(
                """
                # v3 cutover

                `release/current_manifest.json` is the authority for version, region,
                and policy bundle. After its cutover date, runtime config and code use
                the selected contracts; older documents remain historical references.
                """
            ).strip()
            + "\n",
            "docs/authority_precedence.md": dedent(
                """
                # Versioned-client evidence precedence

                The current release manifest selects the active API, region, and
                policy bundle. The selected policy bundle and current API/serializer
                contracts outrank archived rollout notes and pre-cutover API docs.
                """
            ).strip()
            + "\n",
            "policies/regional_override.md": dedent(
                """
                # Regional warehouse override

                Policy bundle `warehouse-policy-2026-03` defines the active values.
                Missing warehouses use `unknown`. For region `eu-west`, normalize
                `ams-old` to `eu-central` and `dub-legacy` to `eu-west`.
                Store aliases by region in runtime configuration and apply the table
                selected by the active manifest region before summary grouping.
                """
            ).strip()
            + "\n",
            "policies/record_resolution.md": dedent(
                """
                # Record resolution

                Canonicalize SKU case, ignore malformed records, and select the
                latest duplicate using timezone-aware timestamp comparison.
                """
            ).strip()
            + "\n",
            "policies/temporal_resolution.md": dedent(
                """
                # Temporal resolution

                Compare ISO-8601 timestamps as timezone-aware instants, not as strings.
                For a normalized SKU, a later instant replaces an earlier record.
                The temporal policy is consumed by record resolution before serialization.
                """
            ).strip()
            + "\n",
            "docs/serializer_contract.md": dedent(
                """
                # Serializer contract

                The serializer emits request_id, next_cursor, record_count,
                total_quantity, and sorted unique warehouses from canonical records.
                """
            ).strip()
            + "\n",
            "release/migration_acceptance.md": dedent(
                """
                # Active v3 migration acceptance record

                This release record is part of the active authority chain selected by
                `current_manifest.json`. Acceptance requires the runtime configuration,
                schema policy, temporal policy, record policy, adapter, and serializer
                to agree on one v3 contract. A repair is incomplete if any artifact
                embeds the archived v2 collection, quantity, or cursor names.

                The required data flow is sequential: load the active configuration;
                select the configured collection; read the configured quantity field;
                canonicalize and resolve records; construct the canonical response; and
                serialize its summary. Configuration overrides supplied by callers use
                the same path and must not be bypassed by hard-coded v3 field names.

                Record resolution is deliberately order-sensitive only at the candidate
                comparison step. Identity is stripped uppercase SKU. Invalid mappings,
                absent SKU, and non-numeric quantities are skipped. Missing or empty
                warehouse values use the configured fallback, then the alias table for
                the configured region is applied. Duplicate timestamps are compared as
                timezone-aware instants and only a strictly later candidate replaces the
                retained record. Serialization occurs after resolution and is stable.
                """
            ).strip()
            + "\n",
            "docs/integration_contract.md": dedent(
                """
                # Cross-artifact integration contract

                The six implementation artifacts have distinct responsibilities.
                `schema_policy.py` selects records and quantities using runtime keys.
                `temporal_policy.py` compares ISO-8601 values as aware instants.
                `record_policy.py` owns validation, canonical identity, aliases, missing
                values, and latest-record selection. `adapter.py` loads configuration and
                composes schema plus record policy. `serializer.py` aggregates only the
                canonical adapter output. `config/client.json` contains the active values.

                No layer may silently reimplement another layer with literal field names.
                An alternate, otherwise-valid configuration can rename the collection,
                quantity, and cursor fields, select another region alias table, and set a
                different missing value. The same functions must continue to work. Output
                keys remain request_id, next_cursor, record_count, total_quantity, and
                warehouses; warehouses are unique and ascending. Empty canonical records
                produce zero counts, zero quantity, and an empty warehouse list.

                Verification should exercise the visible sample twice and compare the
                serialized values, not source text. A deterministic but semantically wrong
                result does not satisfy the contract. Archived v2 notes remain useful only
                for recognizing stale literals and never override this active integration
                record or the manifest-selected policies.
                """
            ).strip()
            + "\n",
            "notes/legacy_rollout.md": "# Archived rollout\n\nA January experiment pinned API v2 globally. It was superseded by the manifest cutover.\n",
        }
    )
    add_distractor_documents(files, rng, 5)
    return _hard_scenario(
        scenario_id="versioned_client_migration",
        title="Versioned Client Migration",
        description="Resolve the active authority chain, update runtime configuration, and repair parser and serializer behavior.",
        profile=profile,
        files=files,
        correct_files=correct_files,
        bugs=bugs,
        parser_module="adapter",
        summary_module="serializer",
        parser_path=parser_path,
        summary_path=summary_path,
        config_path=config_path,
        support_paths=[schema_path, temporal_path, policy_path],
    )


def _hard_scenario(
    *,
    scenario_id: str,
    title: str,
    description: str,
    profile: D5Profile,
    files: dict[str, str],
    correct_files: dict[str, str],
    bugs: list[dict[str, object]],
    parser_module: str,
    summary_module: str,
    parser_path: str,
    summary_path: str,
    config_path: str,
    support_paths: list[str] | None = None,
) -> dict[str, object]:
    labels = [str(bug["label"]) for bug in bugs]
    support_paths = list(support_paths or [])
    authority_inputs = (
        [
            "release/current_manifest.json",
            "docs/api_response_v3.md",
            "docs/authority_precedence.md",
            "policies/warehouse-policy-2026-03.md",
        ]
        if scenario_id == "client_adapter_policy_sync"
        else [
            "release/current_manifest.json",
            "release/migration_acceptance.md",
            "docs/api_v3.md",
            "docs/authority_precedence.md",
            "docs/integration_contract.md",
            "policies/regional_override.md",
            "policies/record_resolution.md",
        ]
    )
    composition_stages = [
        {
            "stage_id": "resolve_authority",
            "required_inputs": authority_inputs,
            "produced_artifacts": [config_path],
            "capability": "authority_resolution",
        }
    ]
    composition_dependencies: list[list[str]] = []
    previous_stage = "resolve_authority"
    if support_paths:
        for index, support_path in enumerate(support_paths):
            if "schema" in support_path:
                stage_id = "resolve_schema_policy"
                evidence_path = (
                    "docs/api_response_v3.md" if scenario_id == "client_adapter_policy_sync" else "docs/api_v3.md"
                )
                capability = "schema_mapping"
            elif "temporal" in support_path:
                stage_id = "resolve_temporal_policy"
                evidence_path = "policies/temporal_resolution.md"
                capability = "timestamp_resolution"
            else:
                stage_id = "resolve_record_policy"
                evidence_path = "policies/record_resolution.md"
                capability = "deduplication"
            required_inputs = [config_path, evidence_path]
            if index:
                required_inputs.extend(support_paths[:index])
            composition_stages.append(
                {
                    "stage_id": stage_id,
                    "required_inputs": required_inputs,
                    "produced_artifacts": [support_path],
                    "capability": capability,
                }
            )
            composition_dependencies.append([previous_stage, stage_id])
            previous_stage = stage_id
    composition_stages.extend(
        [
            {
                "stage_id": "normalize_response",
                "required_inputs": [config_path, *support_paths],
                "produced_artifacts": [parser_path],
                "capability": "schema_mapping",
            },
            {
                "stage_id": "serialize_summary",
                "required_inputs": [parser_path],
                "produced_artifacts": [summary_path],
                "capability": "output_contract",
            },
        ]
    )
    composition_dependencies.extend(
        [
            [previous_stage, "normalize_response"],
            ["normalize_response", "serialize_summary"],
        ]
    )
    required_artifacts = [config_path, *support_paths, parser_path, summary_path]
    return {
        "scenario_id": scenario_id,
        "title": title,
        "description": description,
        "target_path": parser_path,
        "task_type": "multi_artifact_evidence_synthesis",
        "output_style": "code_and_config_patch",
        "entrypoint": "python run_example.py",
        "files": files,
        "expected_output": None,
        "reference_solution_files": correct_files,
        "correct_files": correct_files,
        "bugs": bugs,
        "partial_solution_lattice_profile": {
            "no_fix_score": 0.15,
            "single_fix_max_score": 0.30,
            "pair_fix_max_score": 0.45,
            "all_but_one_max_score": 0.85,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "defect_bundle": {
            "bundle_id": f"retrieval_{profile.profile_id}_evidence_program_chain",
            "defect_ids": labels,
            "dependency_edges": [
                ["authority_config", "quantity_parsing"],
                ["quantity_parsing", "missing_value_policy"],
                ["authority_config", "regional_override"],
                ["deduplication", "timestamp_resolution"],
                ["timestamp_resolution", "output_contract"],
            ],
            "capability_groups": {
                "authority_resolution": ["authority_config"],
                "schema_mapping": ["schema_mapping" if "schema_mapping" in labels else "authority_config"],
                "quantity_parsing": ["quantity_parsing"],
                "pagination": ["pagination"],
                "missing_value_policy": ["missing_value_policy"],
                "regional_override": ["regional_override"],
                "deduplication": ["deduplication"],
                "timestamp_resolution": ["timestamp_resolution"],
                "output_contract": ["output_contract"],
                "hidden_generalization": labels,
            },
            "required_files": required_artifacts,
            "semantic_dependency_depth": profile.semantic_dependency_depth,
        },
        "evaluator_config": {
            "mode": "hidden_tests",
            "runner": "run_hidden_tests.py",
            "target_path": parser_path,
        },
        "hidden_json_assets": {},
        "hidden_text_assets": {
            "run_hidden_tests.py": _hidden_runner(
                parser_module=parser_module,
                summary_module=summary_module,
                config_path=config_path,
            )
        },
        "hints": [
            "Determine the active authority before changing configuration or code.",
            "Interface, pagination, warehouse, identity, and output contracts live in separate documents.",
            "Run the visible example, but expect hidden payload and configuration variants.",
        ],
        "output_contract": [
            "Update " + ", ".join(f"`{path}`" for path in required_artifacts) + " consistently.",
            "Preserve normalize_response() and build_summary() public interfaces.",
            "Runtime behavior must derive from configuration rather than hard-coded visible values.",
        ],
        "structure": base_profile(
            files=files,
            difficulty=5,
            task_type="multi_artifact_evidence_synthesis",
            output_style="code_and_config_patch",
            repair_surface="distributed_contract_and_multi_file_repair",
            failure_mode="authority_dependency_and_semantic_integration",
            smoke_test_quality="partial_non_revealing",
            content_variant_id=scenario_id,
            d5_profile=profile.profile_id,
            semantic_dependency_depth=profile.semantic_dependency_depth,
            capability_count=len(CAPABILITY_WEIGHTS),
            solution_artifact_count=len(required_artifacts),
        ),
        "composition_spec": {
            "stages": composition_stages,
            "dependencies": composition_dependencies,
            "stage_count": len(composition_stages),
            "downstream_consumes_upstream_artifact": True,
        },
        "document_roots": document_roots(files),
    }
