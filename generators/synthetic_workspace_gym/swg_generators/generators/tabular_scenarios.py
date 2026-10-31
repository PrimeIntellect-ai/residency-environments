from __future__ import annotations

import csv
import io
import json
import random
from datetime import date, datetime, timedelta

from swg_generators.schemas import EnvironmentSpec

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y")
TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M")


def build_monthly_segment_report_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    segments = ("enterprise", "midmarket", "smb")
    regions = ("north", "south", "east", "west")
    statuses = ("completed", "processing", "cancelled")
    include_customers = spec.difficulty >= 3
    include_adjustments = spec.difficulty >= 4
    include_duplicates = spec.difficulty >= 2

    customers = [
        {
            "customer_id": f"C{index + 1:03d}",
            "segment": segments[index % len(segments)],
            "region": regions[(index + 1) % len(regions)],
        }
        for index in range(4 + spec.difficulty)
    ]
    rng.shuffle(customers)

    base_date = date(2024, 1, 5)
    orders: list[dict[str, str]] = []
    for index in range(10 + (spec.difficulty * 5)):
        customer = rng.choice(customers)
        order_date = base_date + timedelta(days=rng.randint(0, 75))
        updated_at = datetime(2024, 3, 1, 8, 0, 0) + timedelta(hours=index)
        amount = round(rng.uniform(75, 900), 2)
        orders.append(
            {
                "order_id": f"O{index + 1:04d}",
                "customer_id": customer["customer_id"],
                "status": rng.choices(statuses, weights=(0.65, 0.2, 0.15), k=1)[0],
                "order_date": order_date.strftime(rng.choice(DATE_FORMATS)),
                "updated_at": updated_at.isoformat(),
                "amount": f"{amount:.2f}",
                "sales_owner": f"rep-{rng.randint(1, 8)}",
                "legacy_code": f"L-{rng.randint(100, 999)}",
            }
        )

    if include_duplicates:
        duplicate_indices = rng.sample(range(len(orders)), k=max(1, len(orders) // 5))
        for duplicate_index in duplicate_indices:
            source = orders[duplicate_index]
            updated_at = datetime.fromisoformat(source["updated_at"]) + timedelta(hours=24)
            revised_amount = round(float(source["amount"]) + rng.uniform(-20, 60), 2)
            orders.append(
                {
                    **source,
                    "updated_at": updated_at.isoformat(),
                    "amount": f"{revised_amount:.2f}",
                }
            )
    rng.shuffle(orders)

    adjustments: list[dict[str, str]] = []
    if include_adjustments:
        unique_order_ids = sorted({row["order_id"] for row in orders})
        selected = rng.sample(unique_order_ids, k=max(2, len(unique_order_ids) // 4))
        adjustments = [
            {"order_id": order_id, "adjustment": f"{round(rng.uniform(-15, 25), 2):.2f}"} for order_id in selected
        ]

    customer_lookup = {row["customer_id"]: row for row in customers}
    adjustment_lookup = {row["order_id"]: float(row["adjustment"]) for row in adjustments}
    iterable = orders
    if include_duplicates:
        deduped: dict[str, dict[str, str]] = {}
        for row in orders:
            current = deduped.get(row["order_id"])
            if current is None or current["updated_at"] < row["updated_at"]:
                deduped[row["order_id"]] = row
        iterable = list(deduped.values())

    grouped: dict[tuple[str, ...], dict[str, str | int | float]] = {}
    for row in iterable:
        if row["status"] == "cancelled":
            continue
        parsed_date = parse_mixed_date(row["order_date"])
        key_parts = [parsed_date.strftime("%Y-%m")]
        if include_customers:
            key_parts.append(customer_lookup[row["customer_id"]]["segment"])
        key = tuple(key_parts)
        if key not in grouped:
            entry: dict[str, str | int | float] = {"month": key_parts[0], "order_count": 0, "total_amount": 0.0}
            if include_customers:
                entry["segment"] = key_parts[1]
            grouped[key] = entry
        amount = float(row["amount"])
        if include_adjustments:
            amount += adjustment_lookup.get(row["order_id"], 0.0)
        grouped[key]["order_count"] = int(grouped[key]["order_count"]) + 1
        grouped[key]["total_amount"] = round(float(grouped[key]["total_amount"]) + amount, 2)

    sort_keys = ["month"] + (["segment"] if include_customers else [])
    expected_output = sorted(
        grouped.values(),
        key=lambda item: tuple(str(item[key]) for key in sort_keys),
    )

    input_files = ["data/orders.csv"]
    operations = ["parse_dates", "filter_cancelled"]
    if include_duplicates:
        operations.append("deduplicate_latest")
    files = {
        "data/orders.csv": csv_text(orders, list(orders[0].keys())),
    }
    if include_customers:
        input_files.append("data/customers.json")
        operations.append("join_customers")
        files["data/customers.json"] = json.dumps(customers, indent=2, sort_keys=True) + "\n"
    if include_adjustments:
        input_files.append("data/adjustments.csv")
        operations.append("apply_adjustments")
        files["data/adjustments.csv"] = csv_text(adjustments, ["order_id", "adjustment"])
    if spec.difficulty >= 4:
        files["notes/legacy_columns.md"] = (
            "The `sales_owner` and `legacy_code` fields are historical artifacts and do not belong in the final report.\n"
        )

    return {
        "scenario_id": "monthly_segment_report",
        "title": "Monthly Segment Report",
        "description": "Build a cleaned monthly report from messy order data with optional deduplication, joins, and adjustments.",
        "output_path": "outputs/report.json",
        "input_files": input_files,
        "operations": operations,
        "output_contract": [
            "Write a JSON array to `outputs/report.json`.",
            f"Group by `{sort_keys[0]}`" + (f" and `{sort_keys[1]}`." if len(sort_keys) > 1 else "."),
            "Include `order_count` and `total_amount` for every group.",
            "Sort rows lexicographically by the grouping keys.",
            "Round `total_amount` to 2 decimal places.",
        ],
        "hints": [
            "Use the latest `updated_at` row when duplicate `order_id` values appear.",
            "Cancelled orders should be excluded from the final report.",
            "Customer segment information only exists in the customer lookup file.",
        ],
        "structure": {
            "task_type": "grouped_report",
            "input_shape": "csv_json_mix",
            "time_bucketing": "monthly",
            "output_style": "sorted_row_summary",
        },
        "files": files,
        "expected_output": expected_output,
        "task_descriptor": {
            "input_files": input_files,
            "output_path": "outputs/report.json",
            "operations": operations,
            "group_by": sort_keys,
            "sort_by": sort_keys,
        },
    }


def build_channel_status_pivot_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    stores = [
        {"store_id": "S-01", "channel": "Retail"},
        {"store_id": "S-02", "channel": "Retail "},
        {"store_id": "S-03", "channel": " online"},
        {"store_id": "S-04", "channel": "Partner"},
        {"store_id": "S-05", "channel": "partner "},
    ]
    statuses = ("completed", "processing", "cancelled")
    orders: list[dict[str, object]] = []
    base_date = datetime(2024, 2, 1, 9, 0)
    for index in range(12 + (spec.difficulty * 3)):
        store = rng.choice(stores)
        stamp = base_date + timedelta(hours=index * 6)
        status = rng.choices(statuses, weights=(0.55, 0.3, 0.15), k=1)[0]
        orders.append(
            {
                "order_id": f"CP-{index + 1:03d}",
                "store_id": store["store_id"],
                "status": status.upper() if index % 2 else status,
                "amount": round(rng.uniform(50, 450), 2),
                "placed_at": stamp.strftime(rng.choice(TIMESTAMP_FORMATS)),
            }
        )

    adjustments: list[dict[str, object]] = []
    if spec.difficulty >= 4:
        selected = rng.sample(orders, k=max(2, len(orders) // 5))
        adjustments = [{"order_id": row["order_id"], "adjustment": round(rng.uniform(-10, 30), 2)} for row in selected]

    channel_lookup = {row["store_id"]: normalize_text(row["channel"]) for row in stores}
    adjustment_lookup = {row["order_id"]: float(row["adjustment"]) for row in adjustments}
    grouped: dict[str, dict[str, object]] = {}
    for row in orders:
        status = normalize_text(str(row["status"]))
        if status == "cancelled":
            continue
        channel = channel_lookup[str(row["store_id"])]
        if channel not in grouped:
            grouped[channel] = {
                "channel": channel,
                "completed_orders": 0,
                "processing_orders": 0,
                "total_amount": 0.0,
            }
        if status == "completed":
            grouped[channel]["completed_orders"] = int(grouped[channel]["completed_orders"]) + 1
        elif status == "processing":
            grouped[channel]["processing_orders"] = int(grouped[channel]["processing_orders"]) + 1
        amount = float(row["amount"]) + adjustment_lookup.get(str(row["order_id"]), 0.0)
        grouped[channel]["total_amount"] = round(float(grouped[channel]["total_amount"]) + amount, 2)

    expected_output = sorted(grouped.values(), key=lambda item: str(item["channel"]))
    files = {
        "data/orders.json": json.dumps(orders, indent=2, sort_keys=True) + "\n",
        "data/store_channels.csv": csv_text(stores, ["store_id", "channel"]),
    }
    input_files = ["data/orders.json", "data/store_channels.csv"]
    operations = ["normalize_channels", "filter_cancelled", "aggregate_status_counts"]
    if adjustments:
        files["data/order_adjustments.json"] = json.dumps(adjustments, indent=2, sort_keys=True) + "\n"
        input_files.append("data/order_adjustments.json")
        operations.append("apply_adjustments")
    if spec.difficulty >= 4:
        files["notes/channel_mapping.md"] = (
            "Whitespace and casing in store channel names are not meaningful; normalize them before aggregation.\n"
        )

    return {
        "scenario_id": "channel_status_pivot",
        "title": "Channel Status Pivot",
        "description": "Normalize store channels, filter cancelled orders, and produce a pivot-style summary by sales channel.",
        "output_path": "outputs/channel_pivot.json",
        "input_files": input_files,
        "operations": operations,
        "output_contract": [
            "Write a JSON array to `outputs/channel_pivot.json`.",
            "Include `channel`, `completed_orders`, `processing_orders`, and `total_amount`.",
            "Normalize channel names before aggregation.",
            "Sort rows by `channel`.",
        ],
        "hints": [
            "Store channels come from the lookup file, not the order rows.",
            "Cancelled orders should not contribute to either counts or amount totals.",
            "Whitespace and casing differences in channel labels are noise.",
        ],
        "structure": {
            "task_type": "pivot_report",
            "input_shape": "json_csv_mix",
            "time_bucketing": "none",
            "output_style": "sorted_row_summary",
        },
        "files": files,
        "expected_output": expected_output,
        "task_descriptor": {
            "input_files": input_files,
            "output_path": "outputs/channel_pivot.json",
            "operations": operations,
        },
    }


def build_weekly_refund_rollup_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    accounts = [
        {"account_id": "A-01", "region": "North"},
        {"account_id": "A-02", "region": "west "},
        {"account_id": "A-03", "region": "EAST"},
        {"account_id": "A-04", "region": "north"},
    ]
    event_types = ("sale", "refund")
    statuses = ("posted", "pending")
    base_time = datetime(2024, 3, 1, 10, 30)
    events: list[dict[str, object]] = []
    for index in range(10 + (spec.difficulty * 3)):
        stamp = base_time + timedelta(hours=index * 18)
        event_type = rng.choices(event_types, weights=(0.7, 0.3), k=1)[0]
        events.append(
            {
                "event_id": f"EV-{index + 1:03d}",
                "account_id": rng.choice(accounts)["account_id"],
                "event_type": event_type.upper() if index % 2 else event_type,
                "status": rng.choices(statuses, weights=(0.8, 0.2), k=1)[0].title(),
                "amount": round(rng.uniform(20, 180), 2),
                "booked_at": stamp.strftime(rng.choice(TIMESTAMP_FORMATS)),
            }
        )

    region_lookup = {row["account_id"]: normalize_text(row["region"]) for row in accounts}
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in events:
        if normalize_text(str(row["status"])) != "posted":
            continue
        booked_at = parse_mixed_timestamp(str(row["booked_at"]))
        iso_year, iso_week, _ = booked_at.isocalendar()
        week = f"{iso_year}-W{iso_week:02d}"
        region = region_lookup[str(row["account_id"])]
        key = (week, region)
        if key not in grouped:
            grouped[key] = {"week": week, "region": region, "event_count": 0, "net_amount": 0.0}
        sign = -1.0 if normalize_text(str(row["event_type"])) == "refund" else 1.0
        grouped[key]["event_count"] = int(grouped[key]["event_count"]) + 1
        grouped[key]["net_amount"] = round(float(grouped[key]["net_amount"]) + (float(row["amount"]) * sign), 2)

    expected_output = sorted(grouped.values(), key=lambda item: (str(item["week"]), str(item["region"])))
    files = {
        "data/events.json": json.dumps(events, indent=2, sort_keys=True) + "\n",
        "data/accounts.csv": csv_text(accounts, ["account_id", "region"]),
    }
    if spec.difficulty >= 4:
        files["notes/week_buckets.md"] = (
            "Bucket rows by ISO week and normalize region strings from the account lookup file.\n"
        )

    return {
        "scenario_id": "weekly_refund_rollup",
        "title": "Weekly Refund Rollup",
        "description": "Parse mixed timestamp formats, join account metadata, and produce a weekly net-amount rollup where refunds subtract from sales.",
        "output_path": "outputs/weekly_rollup.json",
        "input_files": ["data/events.json", "data/accounts.csv"],
        "operations": ["parse_timestamps", "join_accounts", "filter_pending", "subtract_refunds", "bucket_by_week"],
        "output_contract": [
            "Write a JSON array to `outputs/weekly_rollup.json`.",
            "Include `week`, `region`, `event_count`, and `net_amount`.",
            "Use ISO-style week buckets like `2024-W09`.",
            "Normalize `region` by stripping whitespace and lowercasing account lookup values; do not title-case regions.",
            "Sort rows by `week` then `region`.",
        ],
        "hints": [
            "Pending events are visible noise and should not affect the rollup.",
            "Refund rows subtract from the weekly total instead of adding to it.",
            "Region names come from the account lookup file and should be emitted in canonical lowercase.",
        ],
        "structure": {
            "task_type": "time_bucket_rollup",
            "input_shape": "json_csv_mix",
            "time_bucketing": "iso_week",
            "region_normalization": "strip_lowercase",
            "output_style": "sorted_row_summary",
        },
        "files": files,
        "expected_output": expected_output,
        "task_descriptor": {
            "input_files": ["data/events.json", "data/accounts.csv"],
            "output_path": "outputs/weekly_rollup.json",
            "operations": ["parse_timestamps", "join_accounts", "filter_pending", "subtract_refunds", "bucket_by_week"],
            "normalization": {"region": "strip_lowercase"},
        },
    }


def build_supplier_restock_summary_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    aliases = {
        "north hub": "north",
        "north": "north",
        "NORTH": "north",
        "south-hub": "south",
        "south": "south",
        "SOUTH": "south",
        "east dc": "east",
        "east": "east",
    }
    canonical_warehouses = ["north", "south", "east"]
    inventory_rows: list[dict[str, object]] = []
    for index in range(8 + (spec.difficulty * 2)):
        canonical = rng.choice(canonical_warehouses)
        alias_candidates = [alias for alias, resolved in aliases.items() if resolved == canonical]
        inventory_rows.append(
            {
                "sku": f"SKU-{index + 1:03d}",
                "warehouse_code": rng.choice(alias_candidates),
                "active": "true" if rng.random() > 0.2 else "false",
                "units_on_hand": rng.randint(4, 40),
            }
        )

    restocks: list[dict[str, object]] = []
    selected_rows = rng.sample(inventory_rows, k=max(3, len(inventory_rows) // 3))
    for row in selected_rows:
        restocks.append(
            {
                "sku": row["sku"],
                "warehouse_code": row["warehouse_code"],
                "pending_units": rng.randint(2, 15),
            }
        )

    pending_lookup = {
        (str(row["sku"]), normalize_text(str(row["warehouse_code"]))): int(row["pending_units"]) for row in restocks
    }
    grouped: dict[str, dict[str, object]] = {}
    for row in inventory_rows:
        if normalize_text(str(row["active"])) != "true":
            continue
        warehouse_key = normalize_text(str(row["warehouse_code"]))
        warehouse = aliases[warehouse_key]
        if warehouse not in grouped:
            grouped[warehouse] = {"warehouse": warehouse, "sku_count": 0, "total_units": 0}
        grouped[warehouse]["sku_count"] = int(grouped[warehouse]["sku_count"]) + 1
        total_units = int(row["units_on_hand"]) + pending_lookup.get((str(row["sku"]), warehouse_key), 0)
        grouped[warehouse]["total_units"] = int(grouped[warehouse]["total_units"]) + total_units

    expected_output = sorted(grouped.values(), key=lambda item: str(item["warehouse"]))
    files = {
        "data/inventory.csv": csv_text(
            inventory_rows,
            ["sku", "warehouse_code", "active", "units_on_hand"],
        ),
        "data/warehouse_aliases.json": json.dumps(aliases, indent=2, sort_keys=True) + "\n",
        "data/restocks.csv": csv_text(restocks, ["sku", "warehouse_code", "pending_units"]),
    }
    if spec.difficulty >= 4:
        files["notes/inactive_items.md"] = (
            "Inactive SKUs are present for realism and should not appear in the summary.\n"
        )

    return {
        "scenario_id": "supplier_restock_summary",
        "title": "Supplier Restock Summary",
        "description": "Normalize warehouse aliases, filter inactive inventory rows, apply pending restocks, and aggregate capacity by warehouse.",
        "output_path": "outputs/restock_summary.json",
        "input_files": ["data/inventory.csv", "data/warehouse_aliases.json", "data/restocks.csv"],
        "operations": ["normalize_aliases", "filter_inactive", "apply_pending_restocks", "aggregate_inventory"],
        "output_contract": [
            "Write a JSON array to `outputs/restock_summary.json`.",
            "Include `warehouse`, `sku_count`, and `total_units`.",
            "Normalize warehouse aliases before joining restock rows.",
            "Sort rows by `warehouse`.",
        ],
        "hints": [
            "Warehouse aliases need normalization before inventory and restock rows can be joined.",
            "Inactive inventory rows are distractors and should be excluded.",
            "Pending restocks add to the warehouse total for the matching SKU.",
        ],
        "structure": {
            "task_type": "normalization_and_join",
            "input_shape": "csv_json_csv",
            "time_bucketing": "none",
            "output_style": "sorted_row_summary",
        },
        "files": files,
        "expected_output": expected_output,
        "task_descriptor": {
            "input_files": ["data/inventory.csv", "data/warehouse_aliases.json", "data/restocks.csv"],
            "output_path": "outputs/restock_summary.json",
            "operations": ["normalize_aliases", "filter_inactive", "apply_pending_restocks", "aggregate_inventory"],
        },
    }


def csv_text(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def normalize_text(value: str) -> str:
    return value.strip().lower()


def parse_mixed_date(value: str) -> datetime:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def parse_mixed_timestamp(value: str) -> datetime:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value}")
