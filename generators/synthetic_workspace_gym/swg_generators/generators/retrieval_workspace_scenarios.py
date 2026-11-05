from __future__ import annotations

import json
import random
from copy import deepcopy
from textwrap import dedent

from swg_generators.generators.d5_profiles import select_d5_profile
from swg_generators.schemas import EnvironmentSpec

DOCUMENT_PREFIXES = (
    "docs/",
    "notes/",
    "specs/",
    "logs/",
    "changelog/",
    "policies/",
    "release/",
)


def render_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def difficulty_settings(difficulty: int) -> dict[str, object]:
    return {
        "retrieval_hops": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty],
        "distractor_count": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}[difficulty],
        "authority_chain_depth": {1: 1, 2: 2, 3: 2, 4: 2, 5: 4}[difficulty],
        "conflict_count": 0 if difficulty <= 3 else 1,
        "conflict_resolution_required": difficulty >= 4,
        "evidence_distribution": (
            "single_source" if difficulty == 1 else "primary_plus_supporting" if difficulty == 2 else "multi_source"
        ),
        "staleness_pattern": "none" if difficulty <= 3 else "stale_note" if difficulty == 4 else "superseded_changelog",
    }


def count_documents(files: dict[str, str]) -> int:
    return sum(1 for path in files if path.startswith(DOCUMENT_PREFIXES))


def document_roots(files: dict[str, str]) -> list[str]:
    roots = {path.split("/", 1)[0] for path in files if path.startswith(DOCUMENT_PREFIXES)}
    return sorted(roots)


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def select_fixture_variant(seed: int, variants: list[dict[str, object]]) -> dict[str, object]:
    return deepcopy(variants[(seed - 1) % len(variants)])


def base_profile(
    *,
    files: dict[str, str],
    difficulty: int,
    task_type: str,
    output_style: str,
    repair_surface: str,
    failure_mode: str,
    smoke_test_quality: str,
    content_variant_id: str | None = None,
    **extra: object,
) -> dict[str, object]:
    settings = difficulty_settings(difficulty)
    profile = {
        "task_type": task_type,
        "document_count": count_documents(files),
        "retrieval_hops": settings["retrieval_hops"],
        "authority_chain_depth": settings["authority_chain_depth"],
        "conflict_count": settings["conflict_count"],
        "conflict_resolution_required": settings["conflict_resolution_required"],
        "evidence_distribution": settings["evidence_distribution"],
        "distractor_count": settings["distractor_count"],
        "staleness_pattern": settings["staleness_pattern"],
        "output_style": output_style,
        "repair_surface": repair_surface,
        "bug_scope": "cross_document",
        "failure_mode": failure_mode,
        "smoke_test_quality": smoke_test_quality,
    }
    if content_variant_id is not None:
        profile["content_variant_id"] = content_variant_id
    profile.update(extra)
    return profile


def add_distractor_documents(files: dict[str, str], rng: random.Random, count: int) -> None:
    distractors = [
        (
            "docs/retention_policy.md",
            "# Retention Policy\n\nAnalytics snapshots are retained for 30 days. This note does not affect the target artifact.\n",
        ),
        (
            "docs/auth_rotation.md",
            "# Auth Rotation\n\nService-to-service tokens rotate every 14 days. No target workspace file needs this value.\n",
        ),
        (
            "notes/archive_window.md",
            "# Archive Window\n\nCold storage compaction runs on Sundays at 02:00 UTC.\n",
        ),
        (
            "logs/heartbeat.log",
            "2026-03-20T10:00:00Z INFO worker-heartbeat ok\n2026-03-20T10:05:00Z INFO worker-heartbeat ok\n",
        ),
        (
            "docs/team_calendar.md",
            "# Team Calendar\n\nQuarter planning is scheduled for April 14.\n",
        ),
        (
            "notes/metrics_backfill.md",
            "# Metrics Backfill\n\nDashboards replay from warehouse snapshots every six hours.\n",
        ),
    ]
    rng.shuffle(distractors)
    added = 0
    for path, content in distractors:
        if path in files:
            continue
        files[path] = content
        added += 1
        if added >= count:
            return


def add_d5_authority_chain(
    files: dict[str, str],
    *,
    bundle_id: str,
    archived_bundle_id: str,
    authoritative_paths: list[str],
    stale_paths: list[str],
) -> None:
    """Make D5 conflicts resolvable only by following an explicit provenance chain."""
    missing = [path for path in authoritative_paths + stale_paths if path not in files]
    if missing:
        raise ValueError(f"authority-chain paths are missing from the scenario: {missing}")

    files["changelog/evidence_index.md"] = (
        dedent(
            f"""
        # Evidence Release Index

        - active evidence bundle: `{bundle_id}`
        - archived evidence bundle: `{archived_bundle_id}`

        Resolve conflicting values using only documents tagged with the active evidence bundle.
        The precedence policy explains how to combine facts split across active documents.
        """
        ).strip()
        + "\n"
    )
    files["docs/evidence_precedence.md"] = (
        dedent(
            """
        # Evidence Precedence

        1. Read `changelog/evidence_index.md` to identify the active evidence bundle.
        2. Prefer contract and specification documents tagged with that bundle over workspace defaults.
        3. Merge non-overlapping facts from active notes and logs only after resolving their bundle tag.
        4. Treat documents tagged with an archived bundle as historical, even when their values look valid.
        """
        ).strip()
        + "\n"
    )

    for path in authoritative_paths:
        files[path] = f"Evidence bundle: `{bundle_id}`\n\n{files[path]}"
    for path in stale_paths:
        files[path] = f"Evidence bundle: `{archived_bundle_id}`\n\n{files[path]}"


def service_config_variants() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "ledger_sync_runtime_v2",
            "service_name": "ledger-sync",
            "base_url": "https://svc.internal/v2",
            "timeout_seconds": 45,
            "retry_attempts": 3,
            "region": "ap-south-1",
            "enable_shadow_mode": True,
            "cohort_limit": 250,
            "legacy": {
                "base_url": "https://svc.internal/v1",
                "timeout_seconds": 30,
                "retry_attempts": 1,
                "region": "us-east-1",
                "enable_shadow_mode": False,
                "cohort_limit": 50,
            },
        },
        {
            "variant_id": "quota_router_runtime_v3",
            "service_name": "quota-router",
            "base_url": "https://quota.internal/v3",
            "timeout_seconds": 60,
            "retry_attempts": 4,
            "region": "eu-west-1",
            "enable_shadow_mode": False,
            "cohort_limit": 120,
            "legacy": {
                "base_url": "https://quota.internal/v2",
                "timeout_seconds": 35,
                "retry_attempts": 2,
                "region": "us-east-2",
                "enable_shadow_mode": True,
                "cohort_limit": 40,
            },
        },
        {
            "variant_id": "report_cache_runtime_v4",
            "service_name": "report-cache",
            "base_url": "https://reports.internal/v4",
            "timeout_seconds": 50,
            "retry_attempts": 2,
            "region": "us-west-2",
            "enable_shadow_mode": True,
            "cohort_limit": 500,
            "legacy": {
                "base_url": "https://reports.internal/v3",
                "timeout_seconds": 40,
                "retry_attempts": 1,
                "region": "us-central-1",
                "enable_shadow_mode": False,
                "cohort_limit": 150,
            },
        },
    ]


def migration_plan_variants() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "customer_subscription_v3",
            "schema_version": "v3",
            "rename_fields": [
                {"from": "customer_id", "to": "account_id"},
                {"from": "plan_code", "to": "subscription_tier"},
            ],
            "drop_fields": ["legacy_score"],
            "backfill_rules": [
                {"source": "profiles.country_code", "target": "country"},
                {"source": "events.last_active_at", "target": "activity.last_seen_at"},
            ],
            "validation_checks": [
                "Row counts remain unchanged after migration.",
                "No null account_id values remain after cutover.",
                "Country values match profiles.country_code for migrated rows.",
            ],
            "legacy_rename_fields": [
                {"from": "customer_id", "to": "member_id"},
                {"from": "plan_code", "to": "plan_code"},
            ],
        },
        {
            "variant_id": "workspace_license_v4",
            "schema_version": "v4",
            "rename_fields": [
                {"from": "org_id", "to": "workspace_id"},
                {"from": "seat_plan", "to": "license_tier"},
            ],
            "drop_fields": ["beta_flag"],
            "backfill_rules": [
                {"source": "accounts.region_code", "target": "region"},
                {"source": "events.last_seen_at", "target": "activity.last_seen_at"},
            ],
            "validation_checks": [
                "Row counts remain unchanged after migration.",
                "No null workspace_id values remain after cutover.",
                "Region values match accounts.region_code for migrated rows.",
            ],
            "legacy_rename_fields": [
                {"from": "org_id", "to": "team_id"},
                {"from": "seat_plan", "to": "seat_plan"},
            ],
        },
        {
            "variant_id": "contract_billing_v5",
            "schema_version": "v5",
            "rename_fields": [
                {"from": "subscription_id", "to": "contract_id"},
                {"from": "plan_name", "to": "plan_slug"},
            ],
            "drop_fields": ["legacy_status"],
            "backfill_rules": [
                {"source": "profiles.locale", "target": "locale"},
                {"source": "invoices.last_paid_at", "target": "billing.last_paid_at"},
            ],
            "validation_checks": [
                "Row counts remain unchanged after migration.",
                "No null contract_id values remain after cutover.",
                "Locale values match profiles.locale for migrated rows.",
            ],
            "legacy_rename_fields": [
                {"from": "subscription_id", "to": "agreement_id"},
                {"from": "plan_name", "to": "plan_name"},
            ],
        },
    ]


def incident_variants() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "incident_workspace_sync_quota_guard",
            "incident_id": "INC-204",
            "service": "workspace-sync",
            "impact": "delayed_exports",
            "priority": "P2",
            "severity": "sev2",
            "cause": "quota_guard_trip",
            "owner": "ops-reliability",
            "legacy_owner": "platform",
            "start_utc": "2026-03-21T07:10:00Z",
            "resolution_utc": "2026-03-21T08:05:00Z",
            "actions": ["increase export worker quota", "replay delayed jobs"],
        },
        {
            "variant_id": "incident_billing_sync_replica_lag",
            "incident_id": "INC-318",
            "service": "billing-sync",
            "impact": "delayed_invoices",
            "priority": "P1",
            "severity": "sev1",
            "cause": "db_replica_lag",
            "owner": "db-operations",
            "legacy_owner": "billing-platform",
            "start_utc": "2026-04-02T11:20:00Z",
            "resolution_utc": "2026-04-02T12:04:00Z",
            "actions": ["fail over read replica", "replay invoice sync"],
        },
        {
            "variant_id": "incident_workspace_ingest_rate_limit",
            "incident_id": "INC-512",
            "service": "workspace-ingest",
            "impact": "slowed_imports",
            "priority": "P3",
            "severity": "sev3",
            "cause": "rate_limit_regression",
            "owner": "pipeline-ops",
            "legacy_owner": "platform",
            "start_utc": "2026-04-11T05:05:00Z",
            "resolution_utc": "2026-04-11T05:49:00Z",
            "actions": ["raise ingest rate limit", "replay delayed imports"],
        },
    ]


def legacy_severity_for(current: str) -> str:
    mapping = {"sev1": "sev2", "sev2": "sev3", "sev3": "sev2"}
    return mapping[current]


def client_adapter_variants() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "adapter_sample_req_204",
            "sample_response": {
                "next_cursor": "cursor-2",
                "records": [
                    {"quantity": 4, "sku": "A-1", "warehouse": "east"},
                    {"quantity": 2, "sku": "B-3"},
                    {"quantity": 5, "sku": "C-7", "warehouse": "west"},
                ],
                "request_id": "req-204",
            },
            "sample_expected": {
                "next_cursor": "cursor-2",
                "record_count": 3,
                "request_id": "req-204",
                "total_quantity": 11,
                "warehouses": ["east", "unknown", "west"],
            },
            "alternate_payload": {
                "request_id": "req-9",
                "next_cursor": None,
                "records": [{"sku": "X", "quantity": 1}],
            },
            "alternate_expected": {
                "next_cursor": None,
                "record_count": 1,
                "request_id": "req-9",
                "total_quantity": 1,
                "warehouses": ["unknown"],
            },
        },
        {
            "variant_id": "adapter_sample_req_318",
            "sample_response": {
                "next_cursor": "cursor-9",
                "records": [
                    {"quantity": 3, "sku": "R-2", "warehouse": "north"},
                    {"quantity": 6, "sku": "S-8", "warehouse": "north"},
                    {"quantity": 1, "sku": "T-4"},
                ],
                "request_id": "req-318",
            },
            "sample_expected": {
                "next_cursor": "cursor-9",
                "record_count": 3,
                "request_id": "req-318",
                "total_quantity": 10,
                "warehouses": ["north", "unknown"],
            },
            "alternate_payload": {
                "request_id": "req-22",
                "next_cursor": "cursor-10",
                "records": [{"sku": "Y", "quantity": 2, "warehouse": "south"}],
            },
            "alternate_expected": {
                "next_cursor": "cursor-10",
                "record_count": 1,
                "request_id": "req-22",
                "total_quantity": 2,
                "warehouses": ["south"],
            },
        },
        {
            "variant_id": "adapter_sample_req_511",
            "sample_response": {
                "next_cursor": None,
                "records": [
                    {"quantity": 7, "sku": "L-1", "warehouse": "central"},
                    {"quantity": 2, "sku": "L-9", "warehouse": "east"},
                ],
                "request_id": "req-511",
            },
            "sample_expected": {
                "next_cursor": None,
                "record_count": 2,
                "request_id": "req-511",
                "total_quantity": 9,
                "warehouses": ["central", "east"],
            },
            "alternate_payload": {
                "request_id": "req-44",
                "next_cursor": None,
                "records": [{"sku": "Z", "quantity": 4}],
            },
            "alternate_expected": {
                "next_cursor": None,
                "record_count": 1,
                "request_id": "req-44",
                "total_quantity": 4,
                "warehouses": ["unknown"],
            },
        },
    ]


def build_migration_expected_output(variant: dict[str, object]) -> dict[str, object]:
    rename_fields = list(variant["rename_fields"])
    backfill_rules = list(variant["backfill_rules"])
    created_fields = unique_strings(
        [str(item["to"]) for item in rename_fields] + [str(item["target"]) for item in backfill_rules]
    )
    rename_targets = ", ".join(str(item["to"]) for item in rename_fields)
    drop_fields = ", ".join(str(item) for item in variant["drop_fields"])
    if len(backfill_rules) == 1:
        backfill_sentence = f"Backfill {backfill_rules[0]['target']} from {backfill_rules[0]['source']}."
    else:
        final_rule = backfill_rules[-1]
        leading_rules = ", ".join(f"{item['target']} from {item['source']}" for item in backfill_rules[:-1])
        backfill_sentence = f"Backfill {leading_rules}, and {final_rule['target']} from {final_rule['source']}."
    ordered_steps = [
        f"Create {variant['schema_version']} columns {', '.join(created_fields)}.",
        backfill_sentence,
        f"Update downstream readers to consume {rename_targets} before cutover.",
        f"Drop {drop_fields} after backfill validation passes.",
    ]
    return {
        "backfill_rules": backfill_rules,
        "drop_fields": list(variant["drop_fields"]),
        "ordered_steps": ordered_steps,
        "rename_fields": rename_fields,
        "schema_version": str(variant["schema_version"]),
        "validation_checks": list(variant["validation_checks"]),
    }


def build_client_adapter_hidden_runner(
    variant: dict[str, object],
    *,
    include_malformed_records: bool = False,
) -> str:
    weights = {
        "collection_field": 0.10,
        "quantity_field": 0.05 if include_malformed_records else 0.10,
        "cursor_field": 0.10,
        "request_id": 0.05,
        "record_count": 0.10,
        "total_quantity": 0.15,
        "warehouse_default": 0.10,
        "warehouse_sorting": 0.10,
        "empty_records": 0.05,
        "hidden_payload_generalization": 0.10 if include_malformed_records else 0.15,
    }
    if include_malformed_records:
        weights.update(
            {
                "non_mapping_record": 0.025,
                "missing_quantity": 0.025,
                "nonnumeric_quantity": 0.025,
                "mixed_page_isolation": 0.025,
            }
        )
    runner = (
        dedent(
            f"""
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path


            CAPABILITY_WEIGHTS = {weights!r}
            ALTERNATE_PAYLOAD = {variant["alternate_payload"]!r}
            ALTERNATE_EXPECTED = {variant["alternate_expected"]!r}


            def main() -> None:
                workspace = Path(sys.argv[1]).resolve()
                sys.path.insert(0, str(workspace / "src"))
                try:
                    from client_adapter import build_summary
                except Exception as exc:
                    print(json.dumps({{"import_error": repr(exc)}}))
                    raise

                def check(callback) -> float:
                    try:
                        return float(bool(callback()))
                    except Exception:
                        return 0.0

                shared = {{
                    "request_id": "shared",
                    "records": [
                        {{"quantity": 2, "count": 91, "warehouse": "west"}},
                        {{"quantity": 5, "count": 92, "warehouse": "east"}},
                    ],
                    "items": [
                        {{"quantity": 2, "count": 91, "warehouse": "west"}},
                        {{"quantity": 5, "count": 92, "warehouse": "east"}},
                    ],
                    "next_cursor": "new-cursor",
                    "cursor": "legacy-cursor",
                }}
                capabilities = {{
                    "collection_field": check(
                        lambda: build_summary({{
                            "request_id": "collection",
                            "records": [],
                            "items": [{{"count": 9, "warehouse": "legacy"}}],
                            "next_cursor": None,
                        }})["record_count"] == 0
                    ),
                    "quantity_field": check(
                        lambda: build_summary(shared)["total_quantity"] == 7
                    ),
                    "cursor_field": check(
                        lambda: build_summary(shared)["next_cursor"] == "new-cursor"
                    ),
                    "request_id": check(
                        lambda: build_summary(shared)["request_id"] == "shared"
                    ),
                    "record_count": check(
                        lambda: build_summary(shared)["record_count"] == 2
                    ),
                    "total_quantity": check(
                        lambda: build_summary(shared)["total_quantity"] == 7
                    ),
                    "warehouse_default": check(
                        lambda: build_summary({{
                            "request_id": "default",
                            "records": [{{"quantity": 1}}],
                            "items": [{{"count": 1}}],
                            "next_cursor": None,
                        }})["warehouses"] == ["unknown"]
                    ),
                    "warehouse_sorting": check(
                        lambda: build_summary({{
                            "request_id": "warehouses",
                            "records": [
                                {{"quantity": 1, "count": 1, "warehouse": "west"}},
                                {{"quantity": 1, "count": 1, "warehouse": "east"}},
                                {{"quantity": 1, "count": 1, "warehouse": "west"}},
                            ],
                            "items": [
                                {{"quantity": 1, "count": 1, "warehouse": "west"}},
                                {{"quantity": 1, "count": 1, "warehouse": "east"}},
                                {{"quantity": 1, "count": 1, "warehouse": "west"}},
                            ],
                            "next_cursor": None,
                        }})["warehouses"] == ["east", "west"]
                    ),
                    "empty_records": check(
                        lambda: build_summary({{
                            "request_id": "empty",
                            "records": [],
                            "items": [],
                            "next_cursor": None,
                        }}) == {{
                            "request_id": "empty",
                            "next_cursor": None,
                            "record_count": 0,
                            "total_quantity": 0,
                            "warehouses": [],
                        }}
                    ),
                    "hidden_payload_generalization": check(
                        lambda: build_summary(ALTERNATE_PAYLOAD) == ALTERNATE_EXPECTED
                    ),
                }}
                passed = sum(capabilities.values())
                payload = {{
                    "success": all(value == 1.0 for value in capabilities.values()),
                    "subscores": {{
                        **{{f"capability_{{name}}": value for name, value in capabilities.items()}},
                        "tests_passed": passed,
                        "tests_total": len(capabilities),
                    }},
                    "capability_weights": CAPABILITY_WEIGHTS,
                    "failure_labels": (
                        [] if all(value == 1.0 for value in capabilities.values())
                        else ["hidden_capabilities_failed"]
                    ),
                    "diagnostics": {{
                        "failed_capabilities": [
                            name for name, value in capabilities.items() if value < 1.0
                        ]
                    }},
                }}
                print(json.dumps(payload, sort_keys=True))


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n"
    )
    if include_malformed_records:
        marker = "    passed = sum(capabilities.values())"
        malformed_check = dedent(
            """
            capabilities["non_mapping_record"] = check(
                lambda: build_summary({
                    "request_id": "non-mapping",
                    "records": [None],
                    "next_cursor": None,
                })["record_count"] == 0
            )
            capabilities["missing_quantity"] = check(
                lambda: build_summary({
                    "request_id": "missing",
                    "records": [{"warehouse": "bad"}],
                    "next_cursor": None,
                })["record_count"] == 0
            )
            capabilities["nonnumeric_quantity"] = check(
                lambda: build_summary({
                    "request_id": "nonnumeric",
                    "records": [{"quantity": "not-numeric", "warehouse": "bad"}],
                    "next_cursor": None,
                })["record_count"] == 0
            )
            capabilities["mixed_page_isolation"] = check(
                lambda: build_summary({
                    "request_id": "mixed",
                    "records": [
                        {"quantity": "not-numeric", "warehouse": "bad"},
                        {"quantity": 4, "warehouse": "good"},
                        None,
                    ],
                    "next_cursor": None,
                }) == {
                    "request_id": "mixed",
                    "next_cursor": None,
                    "record_count": 1,
                    "total_quantity": 4,
                    "warehouses": ["good"],
                }
            )
            """
        ).strip()
        indented_check = "    " + malformed_check.replace("\n", "\n    ")
        runner = runner.replace(marker, indented_check + "\n" + marker, 1)
    return runner


def build_service_config_reconciliation_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    variant = select_fixture_variant(spec.seed, service_config_variants())
    expected_config = {
        "base_url": str(variant["base_url"]),
        "cohort_limit": int(variant["cohort_limit"]),
        "enable_shadow_mode": bool(variant["enable_shadow_mode"]),
        "region": str(variant["region"]),
        "retry_attempts": int(variant["retry_attempts"]),
        "service_name": str(variant["service_name"]),
        "timeout_seconds": int(variant["timeout_seconds"]),
    }
    legacy = dict(variant["legacy"])
    visible_files: dict[str, str] = {
        "config/service_config.json": render_json(
            {
                "base_url": legacy["base_url"],
                "cohort_limit": legacy["cohort_limit"],
                "enable_shadow_mode": legacy["enable_shadow_mode"],
                "region": legacy["region"],
                "retry_attempts": legacy["retry_attempts"],
                "service_name": expected_config["service_name"],
                "timeout_seconds": legacy["timeout_seconds"],
            }
        )
    }
    shadow_flag = json.dumps(expected_config["enable_shadow_mode"])
    legacy_shadow_flag = json.dumps(legacy["enable_shadow_mode"])
    rollout_sentence = (
        "Shadow mode is enabled for the beta cohort."
        if expected_config["enable_shadow_mode"]
        else "Shadow mode remains disabled for the current cohort."
    )
    if difficulty == 1:
        visible_files["specs/runtime_contract.md"] = (
            dedent(
                f"""
            # Runtime Contract

            The `config/service_config.json` file must contain:

            - `service_name`: `{expected_config["service_name"]}`
            - `base_url`: `{expected_config["base_url"]}`
            - `timeout_seconds`: `{expected_config["timeout_seconds"]}`
            - `retry_attempts`: `{expected_config["retry_attempts"]}`
            - `region`: `{expected_config["region"]}`
            - `enable_shadow_mode`: `{shadow_flag}`
            - `cohort_limit`: `{expected_config["cohort_limit"]}`
            """
            ).strip()
            + "\n"
        )
    else:
        visible_files["specs/runtime_contract.md"] = (
            dedent(
                f"""
            # Runtime Contract

            The service contract now uses:

            - `service_name`: `{expected_config["service_name"]}`
            - `base_url`: `{expected_config["base_url"]}`
            - `timeout_seconds`: `{expected_config["timeout_seconds"]}`
            - `retry_attempts`: `{expected_config["retry_attempts"]}`
            """
            ).strip()
            + "\n"
        )
        visible_files["notes/rollout_plan.md"] = (
            dedent(
                f"""
            # Rollout Plan

            {rollout_sentence}

            - `enable_shadow_mode`: `{shadow_flag}`
            - `cohort_limit`: `{expected_config["cohort_limit"]}`
            """
            ).strip()
            + "\n"
        )
        if difficulty >= 2:
            visible_files["notes/region_override.md"] = (
                dedent(
                    f"""
                # Region Override

                The current deployment stays pinned to `{expected_config["region"]}` until the next failover rehearsal.
                """
                ).strip()
                + "\n"
            )
    if difficulty == 4:
        visible_files["notes/legacy_service_config.md"] = (
            dedent(
                f"""
            # Legacy Service Config

            Old rollout draft:

            - `base_url`: `{legacy["base_url"]}`
            - `timeout_seconds`: `{legacy["timeout_seconds"]}`
            - `region`: `{legacy["region"]}`
            """
            ).strip()
            + "\n"
        )
    if difficulty == 5:
        visible_files["changelog/2026-01-runtime-rollout.md"] = (
            dedent(
                f"""
            # 2026-01 Runtime Rollout

            Candidate rollout values:

            Historical values:

            - `base_url`: `{legacy["base_url"]}`
            - `enable_shadow_mode`: `{legacy_shadow_flag}`
            - `cohort_limit`: `{legacy["cohort_limit"]}`
            """
            ).strip()
            + "\n"
        )
        add_d5_authority_chain(
            visible_files,
            bundle_id="runtime-2026-03",
            archived_bundle_id="runtime-2026-01",
            authoritative_paths=[
                "specs/runtime_contract.md",
                "notes/rollout_plan.md",
                "notes/region_override.md",
            ],
            stale_paths=["changelog/2026-01-runtime-rollout.md"],
        )
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    output_contract = [
        "Update `config/service_config.json` in place.",
        "Keep `service_name` unchanged.",
        "The final file must be valid JSON with the required values from the document set.",
    ]
    return {
        "scenario_id": "service_config_reconciliation",
        "title": "Service Config Reconciliation",
        "description": "The runtime config has drifted from the local document set. Reconcile the config using only the visible documents in the workspace.",
        "target_path": "config/service_config.json",
        "task_type": "config_alignment",
        "output_style": "config_patch",
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_config,
        "reference_solution_files": {
            "config/service_config.json": render_json(expected_config),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "config/service_config.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The runtime contract is authoritative for transport settings.",
            "Rollout notes may define feature flags separately from the base config.",
            "Ignore clearly stale legacy notes when newer documents supersede them.",
        ],
        "output_contract": output_contract,
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="config_alignment",
            output_style="config_patch",
            repair_surface="config_alignment",
            failure_mode="grounding_and_semantic",
            smoke_test_quality="none",
            content_variant_id=str(variant["variant_id"]),
        ),
        "document_roots": document_roots(visible_files),
    }


def build_migration_plan_bundle_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    variant = select_fixture_variant(spec.seed, migration_plan_variants())
    expected_output = build_migration_expected_output(variant)
    schema_version = str(variant["schema_version"])
    schema_spec_path = f"specs/schema_{schema_version}.md"
    rename_fields = list(variant["rename_fields"])
    backfill_rules = list(variant["backfill_rules"])
    visible_files: dict[str, str] = {
        "artifacts/migration_plan.json": render_json(
            {
                "backfill_rules": [],
                "drop_fields": list(variant["drop_fields"]),
                "ordered_steps": ["Create new columns."],
                "rename_fields": [
                    {
                        "from": str(rename_fields[0]["from"]),
                        "to": str(rename_fields[0]["from"]),
                    }
                ],
                "schema_version": f"{variant['schema_version']}-draft",
                "validation_checks": [],
            }
        )
    }
    if difficulty == 1:
        rename_lines = "\n".join(f"- rename `{item['from']} -> {item['to']}`" for item in rename_fields)
        drop_lines = "\n".join(f"- drop `{item}`" for item in variant["drop_fields"])
        backfill_lines = "\n".join(f"- backfill `{item['target']}` from `{item['source']}`" for item in backfill_rules)
        validation_lines = "\n".join(f"- {item}" for item in variant["validation_checks"])
        visible_files[schema_spec_path] = (
            dedent(
                f"""
            # Schema {schema_version}

            Produce a migration plan with:

            - schema version `{schema_version}`
            {rename_lines}
            {drop_lines}
            {backfill_lines}
            {validation_lines}
            """
            ).strip()
            + "\n"
        )
    else:
        rename_lines = "\n".join(f"- rename `{item['from']} -> {item['to']}`" for item in rename_fields)
        drop_lines = "\n".join(f"- drop `{item}`" for item in variant["drop_fields"])
        visible_files[schema_spec_path] = (
            dedent(
                f"""
            # Schema {schema_version}

            Required schema changes:

            {rename_lines}
            {drop_lines}
            - schema version `{schema_version}`
            """
            ).strip()
            + "\n"
        )
        visible_files["notes/backfill_rules.md"] = (
            dedent(
                "\n".join(
                    [
                        "# Backfill Rules",
                        "",
                        *[f"- `{item['source']} -> {item['target']}`" for item in backfill_rules],
                    ]
                )
            ).strip()
            + "\n"
        )
    cutover_lines = "\n".join(
        f"{index}. {step}" for index, step in enumerate(expected_output["ordered_steps"], start=1)
    )
    validation_lines = "\n".join(f"- {item}" for item in variant["validation_checks"])
    visible_files["docs/client_cutover.md"] = (
        dedent(
            f"""
        # Client Cutover

        Recommended cutover sequence:

        {cutover_lines}

        Validation checks:

        {validation_lines}
        """
        ).strip()
        + "\n"
    )
    if difficulty == 4:
        visible_files["notes/v2_cutover.md"] = (
            dedent(
                "\n".join(
                    [
                        "# v2 Cutover Draft",
                        "",
                        "Old plan draft:",
                        "",
                        *[f"- rename `{item['from']} -> {item['to']}`" for item in variant["legacy_rename_fields"]],
                        f"- keep `{backfill_rules[0]['target']}` empty",
                    ]
                )
            ).strip()
            + "\n"
        )
    if difficulty == 5:
        visible_files["changelog/2026-02-migration.md"] = (
            dedent(
                "\n".join(
                    [
                        f"# 2026-02 Migration for {schema_version}",
                        "",
                        f"Candidate migration values for schema {schema_version}.",
                        "",
                        "Historical draft:",
                        "",
                        *[f"- rename `{item['from']} -> {item['to']}`" for item in variant["legacy_rename_fields"]],
                    ]
                )
            ).strip()
            + "\n"
        )
        add_d5_authority_chain(
            visible_files,
            bundle_id=f"migration-{schema_version}-current",
            archived_bundle_id="migration-v2-draft",
            authoritative_paths=[
                schema_spec_path,
                "notes/backfill_rules.md",
                "docs/client_cutover.md",
            ],
            stale_paths=["changelog/2026-02-migration.md"],
        )
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    return {
        "scenario_id": "migration_plan_bundle",
        "title": "Migration Plan Bundle",
        "description": "The workspace includes migration notes, compatibility guidance, and schema docs. Synthesize them into the required migration plan artifact.",
        "target_path": "artifacts/migration_plan.json",
        "task_type": "migration_planning",
        "output_style": "migration_plan",
        "schema_version": schema_version,
        "schema_spec_path": schema_spec_path,
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_output,
        "reference_solution_files": {
            "artifacts/migration_plan.json": render_json(expected_output),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "artifacts/migration_plan.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The schema spec defines field renames and removals.",
            "Backfill rules may be split away from the cutover order.",
            f"Older migration drafts can be wrong once schema {schema_version} is introduced.",
        ],
        "output_contract": [
            "Write the final plan to `artifacts/migration_plan.json`.",
            "Keep the JSON keys and list ordering stable.",
            f"The plan must reflect the current schema `{schema_version}` from `{schema_spec_path}`, not older migration drafts.",
        ],
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="migration_planning",
            output_style="migration_plan",
            repair_surface="migration_spec_synthesis",
            failure_mode="grounding_and_semantic",
            smoke_test_quality="none",
            content_variant_id=str(variant["variant_id"]),
            schema_version=schema_version,
        ),
        "document_roots": document_roots(visible_files),
    }


def build_incident_report_bundle_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    variant = select_fixture_variant(spec.seed, incident_variants())
    expected_output = {
        "actions": list(variant["actions"]),
        "customer_impact": str(variant["impact"]),
        "incident_id": str(variant["incident_id"]),
        "owner": str(variant["owner"]),
        "primary_cause": str(variant["cause"]),
        "resolution_utc": str(variant["resolution_utc"]),
        "service": str(variant["service"]),
        "severity": str(variant["severity"]),
        "start_utc": str(variant["start_utc"]),
    }
    log_lines = [
        f"{variant['start_utc']} incident={variant['incident_id']} service={variant['service']} event=start impact={variant['impact']} priority={variant['priority']}",
        f"{variant['start_utc']} incident={variant['incident_id']} event=diagnosis marker={variant['cause']}",
    ]
    for offset, action in enumerate(variant["actions"], start=1):
        minute = 12 + (offset * 14)
        timestamp_prefix = str(variant["start_utc"])[:14]
        action_timestamp = f"{timestamp_prefix}{minute:02d}:00Z"
        log_lines.append(f"{action_timestamp} incident={variant['incident_id']} event=action detail={action}")
    log_lines.append(
        f"{variant['resolution_utc']} incident={variant['incident_id']} event=resolved owner_alias={variant['owner']}"
    )
    visible_files: dict[str, str] = {
        "artifacts/oncall_report.json": render_json(
            {
                "actions": [],
                "customer_impact": "unknown",
                "incident_id": variant["incident_id"],
                "owner": variant["legacy_owner"],
                "primary_cause": "unknown",
                "resolution_utc": "",
                "service": variant["service"],
                "severity": legacy_severity_for(str(variant["severity"])),
                "start_utc": "",
            }
        ),
        f"logs/{str(variant['incident_id']).lower()}.log": "\n".join(log_lines) + "\n",
        "specs/report_contract.md": dedent(
            """
            # Report Contract

            The final artifact must be JSON with these keys:

            - `incident_id`
            - `service`
            - `severity`
            - `owner`
            - `customer_impact`
            - `start_utc`
            - `resolution_utc`
            - `primary_cause`
            - `actions`
            """
        ).strip()
        + "\n",
    }
    if difficulty >= 1:
        visible_files["notes/severity_policy.md"] = (
            dedent(
                """
            # Severity Policy

            - `P1 -> sev1`
            - `P2 -> sev2`
            - `P3 -> sev3`
            """
            ).strip()
            + "\n"
        )
    if difficulty >= 3:
        visible_files["docs/action_capture.md"] = (
            dedent(
                """
            # Action Capture

            Every `event=action detail=...` line in the incident log must be preserved in order in the final report.
            The `owner_alias` on the resolved line is the report owner.
            """
            ).strip()
            + "\n"
        )
    if difficulty == 4:
        visible_files["notes/postmortem_draft.md"] = (
            dedent(
                f"""
            # Postmortem Draft

            Draft notes from another incident:

            - severity: {legacy_severity_for(str(variant["severity"]))}
            - owner: {variant["legacy_owner"]}
            """
            ).strip()
            + "\n"
        )
    if difficulty == 5:
        visible_files["changelog/2026-02-reporting.md"] = (
            dedent(
                f"""
            # 2026-02 Reporting

            Candidate severity mapping:

            - `{variant["priority"]} -> {legacy_severity_for(str(variant["severity"]))}`
            """
            ).strip()
            + "\n"
        )
        add_d5_authority_chain(
            visible_files,
            bundle_id="incident-reporting-2026-03",
            archived_bundle_id="incident-reporting-2026-02",
            authoritative_paths=[
                "specs/report_contract.md",
                "notes/severity_policy.md",
                "docs/action_capture.md",
                f"logs/{str(variant['incident_id']).lower()}.log",
            ],
            stale_paths=["changelog/2026-02-reporting.md"],
        )
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    structure = base_profile(
        files=visible_files,
        difficulty=difficulty,
        task_type="incident_reporting",
        output_style="structured_json",
        repair_surface="incident_report_grounding",
        failure_mode="grounding_and_semantic",
        smoke_test_quality="none",
        content_variant_id=str(variant["variant_id"]),
    )
    if difficulty == 1:
        structure["evidence_distribution"] = "primary_plus_supporting"
        structure["retrieval_hops"] = 2
    return {
        "scenario_id": "incident_report_bundle",
        "title": "Incident Report Bundle",
        "description": "Use the local logs, contract notes, and operational guidance to build the required on-call incident report artifact.",
        "target_path": "artifacts/oncall_report.json",
        "task_type": "incident_reporting",
        "output_style": "structured_json",
        "entrypoint": None,
        "files": visible_files,
        "expected_output": expected_output,
        "reference_solution_files": {
            "artifacts/oncall_report.json": render_json(expected_output),
        },
        "evaluator_config": {
            "mode": "exact_json",
            "output_path": "artifacts/oncall_report.json",
            "expected_path": "expected_output.json",
        },
        "hints": [
            "The log contains the authoritative timestamps, root-cause marker, and ordered actions.",
            "Severity translation may live outside the report schema.",
            "Ignore stale drafts that describe a different severity or owner.",
        ],
        "output_contract": [
            "Write the final artifact to `artifacts/oncall_report.json`.",
            "Preserve ordered actions from the log.",
            "Use the visible document set to ground severity and ownership.",
        ],
        "structure": structure,
        "document_roots": document_roots(visible_files),
    }


def build_client_adapter_sync_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    difficulty = spec.difficulty
    profile = select_d5_profile(difficulty, spec.seed)
    variant = select_fixture_variant(spec.seed, client_adapter_variants())
    correct_adapter = (
        dedent(
            """
        from __future__ import annotations


        def build_summary(response: dict[str, object]) -> dict[str, object]:
            records = list(response.get("records", []))
            total_quantity = 0
            warehouses: set[str] = set()
            for record in records:
                total_quantity += int(record["quantity"])
                warehouses.add(str(record.get("warehouse", "unknown")))
            return {
                "request_id": str(response["request_id"]),
                "next_cursor": response.get("next_cursor"),
                "record_count": len(records),
                "total_quantity": total_quantity,
                "warehouses": sorted(warehouses),
            }
        """
        ).strip()
        + "\n"
    )
    buggy_adapter = (
        dedent(
            """
        from __future__ import annotations


        def build_summary(response: dict[str, object]) -> dict[str, object]:
            items = list(response.get("items", []))
            total_quantity = 0
            warehouses: set[str] = set()
            for item in items:
                total_quantity += int(item["count"])
                warehouses.add(str(item["warehouse"]))
            return {
                "request_id": str(response["request_id"]),
                "next_cursor": response.get("cursor"),
                "record_count": len(items),
                "total_quantity": total_quantity,
                "warehouses": sorted(warehouses),
            }
        """
        ).strip()
        + "\n"
    )
    visible_files: dict[str, str] = {
        "src/client_adapter.py": buggy_adapter,
        "samples/response.json": render_json(variant["sample_response"]),
        "run_example.py": dedent(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            workspace = Path(__file__).resolve().parent
            sys.path.insert(0, str(workspace / "src"))

            from client_adapter import build_summary


            def main() -> None:
                payload = json.loads((workspace / "samples" / "response.json").read_text(encoding="utf-8"))
                print(json.dumps(build_summary(payload), indent=2, sort_keys=True))


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        "docs/api_reference.md": dedent(
            """
            # API Reference

            Latest response payload:

            - `request_id`: string
            - `records`: list of objects
            - each record includes `sku`, `quantity`, and optional `warehouse`
            - pagination field: `next_cursor`
            """
        ).strip()
        + "\n",
    }
    if difficulty >= 2:
        visible_files["docs/output_contract.md"] = (
            dedent(
                """
            # Output Contract

            `build_summary()` must return:

            - `request_id`
            - `next_cursor`
            - `record_count`
            - `total_quantity`
            - `warehouses` sorted ascending
            """
            ).strip()
            + "\n"
        )
    if difficulty >= 3:
        visible_files["notes/api_changelog.md"] = (
            dedent(
                """
            # API Changelog

            Renames in the current release:

            - `items -> records`
            - `count -> quantity`
            - `cursor -> next_cursor`

            Missing `warehouse` values should default to `unknown`.
            """
            ).strip()
            + "\n"
        )
    if difficulty == 4:
        visible_files["notes/legacy_adapter.md"] = (
            dedent(
                """
            # Legacy Adapter

            Old adapter behavior:

            - read `items`
            - read `count`
            - fail if `warehouse` is missing
            """
            ).strip()
            + "\n"
        )
    if difficulty == 5:
        visible_files["changelog/2026-01-client.md"] = (
            dedent(
                """
            # 2026-01 Client Notes

            Candidate client response shape:

            - pagination field `cursor`
            - collection field `items`
            """
            ).strip()
            + "\n"
        )
        add_d5_authority_chain(
            visible_files,
            bundle_id="client-api-2026-03",
            archived_bundle_id="client-api-2026-01",
            authoritative_paths=[
                "docs/api_reference.md",
                "docs/output_contract.md",
                "notes/api_changelog.md",
            ],
            stale_paths=["changelog/2026-01-client.md"],
        )
        if profile is None:
            raise ValueError("D5 client adapter scenario requires a D5 profile")
        if profile.profile_id == "d5_a":
            visible_files["docs/current_api_summary.md"] = (
                "# Current client summary\n\n"
                "The active response uses records, per-record quantity, and "
                "next_cursor. Missing warehouses become unknown. Preserve "
                "request_id, count records, total quantities, and return sorted "
                "unique warehouses.\n"
            )
            visible_files["public_check.py"] = (
                dedent(
                    """
                import json
                import sys
                from pathlib import Path
                workspace = Path(__file__).resolve().parent
                sys.path.insert(0, str(workspace / "src"))
                from client_adapter import build_summary
                payload = json.loads((workspace / "samples/response.json").read_text(encoding="utf-8"))
                result = build_summary(payload)
                assert set(result) == {"request_id", "next_cursor", "record_count", "total_quantity", "warehouses"}
                assert result["warehouses"] == sorted(set(result["warehouses"]))
                print("public structure check passed")
                """
                ).strip()
                + "\n"
            )
        elif profile.profile_id == "d5_b":
            visible_files["docs/validation_guide.md"] = (
                "# Validation guide\n\n"
                "Validate the visible sample, an empty records list, and a "
                "record without warehouse. Do not infer values from legacy items.\n"
            )
        else:
            visible_files["docs/version_applicability.md"] = (
                "# Version applicability\n\nThe authority index selects client API bundle client-api-2026-03.\n"
            )
            visible_files["docs/field_mapping.md"] = (
                "# Active field mapping\n\n"
                "items became records, count became quantity, and cursor "
                "became next_cursor.\n"
            )
            visible_files["docs/warehouse_override.md"] = (
                "# Warehouse exception\n\n"
                "For this client version, an omitted warehouse is represented as "
                "unknown; warehouse values are unique and sorted in the summary.\n"
            )
    add_distractor_documents(visible_files, rng, int(difficulty_settings(difficulty)["distractor_count"]))
    defect_specs = [
        (
            "legacy_collection_field",
            'response.get("records", [])',
            'response.get("items", [])',
        ),
        ("legacy_quantity_field", 'record["quantity"]', 'record["count"]'),
        (
            "legacy_cursor_field",
            'response.get("next_cursor")',
            'response.get("cursor")',
        ),
        (
            "missing_warehouse_default",
            'record.get("warehouse", "unknown")',
            'record["warehouse"]',
        ),
    ]
    bugs: list[dict[str, object]] = []
    for label, old, new in defect_specs:

        def apply(content: str, *, old: str = old, new: str = new, label: str = label) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(f"retrieval defect {label!r} did not modify client_adapter.py")
            return updated

        bugs.append({"label": label, "target_path": "src/client_adapter.py", "apply": apply})
    contract = {
        "collection_field": "records",
        "quantity_field": "quantity",
        "cursor_field": "next_cursor",
        "warehouse_default": "unknown",
    }
    reference_solution_files = {"src/client_adapter.py": correct_adapter}
    correct_files = {"src/client_adapter.py": correct_adapter}
    hidden_json_assets: dict[str, object] = {}
    evaluator_config: dict[str, object] = {
        "mode": "hidden_tests",
        "runner": "run_hidden_tests.py",
        "target_path": "src/client_adapter.py",
    }
    if difficulty < 5:
        evaluator_config.update(
            {
                "initial_score": 0.30,
                "score_normalization": "initial_workspace_progress_v1",
            }
        )
    if difficulty == 5:
        stale_contract = {
            "collection_field": "items",
            "quantity_field": "count",
            "cursor_field": "cursor",
            "warehouse_default": "error",
        }
        visible_files["config/adapter_contract.json"] = render_json(stale_contract)
        reference_solution_files["config/adapter_contract.json"] = render_json(contract)
        correct_files["config/adapter_contract.json"] = render_json(contract)
        hidden_json_assets["expected_adapter_contract.json"] = contract

    hidden_runner = build_client_adapter_hidden_runner(variant)
    return {
        "scenario_id": "client_adapter_sync",
        "title": "Client Adapter Sync",
        "description": "Repair the adapter implementation so it aligns with the local API documents and examples.",
        "target_path": "src/client_adapter.py",
        "task_type": "spec_to_code_alignment",
        "output_style": "code_patch",
        "entrypoint": "python run_example.py",
        "files": visible_files,
        "expected_output": None,
        "reference_solution_files": reference_solution_files,
        "correct_files": correct_files,
        "bugs": bugs,
        "partial_solution_lattice_profile": {
            "no_fix_score": 0.0,
            "single_fix_max_score": 0.20,
            "pair_fix_max_score": 0.60,
            "all_but_one_max_score": 0.80,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "defect_bundle": {
            "bundle_id": "api_evidence_adapter_chain",
            "defect_ids": [label for label, _, _ in defect_specs],
            "dependency_edges": [
                ["legacy_collection_field", "legacy_quantity_field"],
                ["legacy_quantity_field", "missing_warehouse_default"],
                ["legacy_collection_field", "legacy_cursor_field"],
            ],
            "capability_groups": {
                "authority_selection": ["legacy_collection_field"],
                "transformation": ["legacy_quantity_field"],
                "cross_file_consistency": ["legacy_cursor_field"],
                "integration": ["missing_warehouse_default"],
            },
            "required_files": [
                "src/client_adapter.py",
                "docs/api_reference.md",
                "docs/output_contract.md",
                "notes/api_changelog.md",
            ],
            "semantic_dependency_depth": 3,
        },
        "evaluator_config": evaluator_config,
        "hidden_json_assets": hidden_json_assets,
        "hidden_text_assets": {
            "run_hidden_tests.py": hidden_runner,
        },
        "hints": [
            "The API reference names the canonical response fields.",
            "The changelog may describe field renames and defaulting behavior.",
            "Use the local sample payload to sanity-check the adapter after patching it.",
        ],
        "output_contract": [
            "Update `src/client_adapter.py` so it matches the visible API docs.",
            "Preserve the public `build_summary()` function signature.",
            "The hidden evaluator checks behavior, not just file text.",
        ],
        "structure": base_profile(
            files=visible_files,
            difficulty=difficulty,
            task_type="spec_to_code_alignment",
            output_style="code_patch",
            repair_surface="doc_assisted_code_repair",
            failure_mode="interface_and_semantic",
            smoke_test_quality="informative",
            content_variant_id=str(variant["variant_id"]),
            d5_profile=(profile.profile_id if profile is not None else None),
            semantic_dependency_depth=(profile.semantic_dependency_depth if profile is not None else 0),
        ),
        "document_roots": document_roots(visible_files),
    }
