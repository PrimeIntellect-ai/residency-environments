"""Report visual decisions separately from completed missions, grouped by family.

Usage: python scripts/evidence.py <label>=<trace file or eval directory> ...
Baselines: add --policy honest|blind_shortcut|broken|always_b --check-baseline
and optionally --expected-count N. A parent containing multiple evals is
ambiguous; select the exact eval directory rather than silently choosing one.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FAMILY_ONE = "land_or_hold"
FAMILY_TWO = "find_pad"
FAMILY_THREE = "obstacle_bypass"
VARIANT_COUNTS = {
    FAMILY_ONE: 2,
    FAMILY_TWO: 2,
    FAMILY_THREE: 2,
    "alternate_pad": 3,
    "patrol_inspection": 2,
    "go_around": 2,
    "compound": 4,
}
PAIRED_FAMILIES = tuple(VARIANT_COUNTS)
ADVANCED_FAMILIES = ("alternate_pad", "patrol_inspection", "go_around", "compound")
POLICIES = ("honest", "blind_shortcut", "broken", "always_b")
RUNTIME_FAILURE_METRICS = ("stalled", "camera_failed", "timed_out_actions", "rejected_replies", "event_failed")


def _trace_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"evidence path does not exist: {path}")
    candidates = sorted(set(path.glob("traces.jsonl")) | set(path.glob("*/traces.jsonl")))
    if len(candidates) != 1:
        found = ", ".join(str(p) for p in candidates) or "none"
        raise ValueError(
            f"select an explicit eval directory or traces.jsonl; found {len(candidates)} trace files in {path}: {found}"
        )
    return candidates[0]


def _number(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _binary(value) -> float | None:
    value = _number(value)
    return value if value in (0.0, 1.0) else None


def _row(episode: dict) -> dict:
    data = episode.get("task", {}).get("data", {})
    family = data.get("family") or "unknown"
    traces = episode.get("traces") or []
    trace = traces[0] if isinstance(traces, list) and len(traces) == 1 and isinstance(traces[0], dict) else None
    error = episode.get("ok") is False or bool(episode.get("errors")) or not trace
    if trace:
        error = error or trace.get("ok") is False or bool(trace.get("errors"))
    trace = trace or {}
    metrics = trace.get("metrics") or {}
    info = trace.get("info") or {}
    reward = (trace.get("rewards") or {}).get("mission")
    score = _number(reward.get("score") if isinstance(reward, dict) else reward)
    if score is not None and not 0.0 <= score <= 1.0:
        score = None
    ending = info.get("ending")
    decision_taken = info.get("decision", ending)
    blocked = data.get("blocked")

    # Other families must define their own decision metric; an action name by
    # itself cannot tell us whether that action was the correct decision.
    if "decision_correct" in metrics:
        decision = _binary(metrics["decision_correct"])
    elif family == FAMILY_ONE and isinstance(blocked, bool) and ("decision" in info or "ending" in info):
        decision = float(decision_taken == ("abort" if blocked else "land"))
    else:
        decision = None

    missing = []
    if score is None:
        missing.append("mission reward")
    if decision is None:
        missing.append("decision_correct")
    # A failed evaluation remains a failed row even if it emitted a partial
    # reward before the error. Missing trace rows stay in the denominator.
    if error:
        score, decision = 0.0, 0.0
    return {
        "family": family,
        "name": data.get("name", "unnamed"),
        "pair": data.get("seed"),
        "variant": data.get("variant"),
        "split": data.get("split", "legacy"),
        "blocked": blocked,
        "score": score,
        "completed": score == 1.0,
        "decision_correct": decision,
        "ending": ending,
        "decision": decision_taken,
        "route_choice": info.get("route_choice"),
        "hit": metrics.get("no_collision") == 0.0,
        "error": bool(error),
        "missing_metrics": tuple(missing),
        "runtime_metrics": {
            key: metrics[key]
            for key in (*RUNTIME_FAILURE_METRICS, "turns", "route_barrier_hit", "scenario_obstacle_hit", "no_collision")
            if key in metrics
        },
    }


def load(outputs_dir: str | Path) -> list[dict]:
    """Read every episode from one unambiguously selected trace file."""
    path = _trace_file(outputs_dir)
    rows = []
    with path.open() as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                if not isinstance(episode, dict):
                    raise ValueError("episode must be an object")
                rows.append(_row(episode))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid episode in {path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no episodes in {path}")
    return rows


def _variant(row: dict) -> int | None:
    """Identify a scene variant within its family's complete counterfactual group."""
    if row.get("family") == FAMILY_ONE and isinstance(row.get("blocked"), bool):
        return int(row["blocked"])
    if (
        row.get("family") in VARIANT_COUNTS
        and row.get("family") != FAMILY_ONE
        and type(row.get("variant")) is int
        and row["variant"] in range(VARIANT_COUNTS[row["family"]])
    ):
        return row["variant"]
    return None


def _pair_key(row: dict) -> tuple | None:
    seed, split = row.get("pair"), row.get("split", "legacy")
    if seed is None or type(seed) not in (str, int) or not isinstance(split, str) or not split:
        return None
    if (
        row.get("family") in VARIANT_COUNTS
        and row.get("family") != FAMILY_ONE
        and split not in ("development", "evaluation")
    ):
        return None
    return row.get("family"), seed, split


def _expected_rows(rows: list[dict], policy: str | None) -> list[float] | None:
    if policy is None:
        return None
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if not rows or any(_variant(row) is None for row in rows):
        return None
    if policy == "always_b":
        if any(row.get("family") != "alternate_pad" for row in rows):
            return None
        return [float(_variant(row) == 1) for row in rows]
    if policy == "blind_shortcut":
        # Fixed controls always land on A (or use the positive branch/corridor).
        # Every family defines that blind shortcut as correct only on variant 0.
        return [float(_variant(row) == 0) for row in rows]
    return [float(policy == "honest")] * len(rows)


def validation_errors(rows: list[dict], policy: str | None = None, expected_count: int | None = None) -> list[str]:
    """Validate report completeness and, optionally, every scripted outcome.

    Hosted runs need complete evidence, not a minimum model reward. Baseline
    acceptance additionally requires unique, complete counterfactual groups
    within each family and split (two, three or four variants).
    """
    failures = []
    if not rows:
        failures.append("no episodes")
    if expected_count is not None and len(rows) != expected_count:
        failures.append(f"expected {expected_count} episodes, found {len(rows)}")
    for index, row in enumerate(rows, start=1):
        label = f"row {index} ({row.get('name', 'unnamed')})"
        if row.get("error"):
            failures.append(f"{label}: evaluation error")
        if row.get("missing_metrics"):
            failures.append(f"{label}: missing/invalid {', '.join(row['missing_metrics'])}")
    if policy is None:
        return failures
    expected = _expected_rows(rows, policy)
    if expected is None:
        return failures + [
            "baseline expectations require supported families and known variants; always_b is only supported for alternate_pad"
        ]
    pairs = defaultdict(list)
    for row in rows:
        key = _pair_key(row)
        if key is None:
            failures.append("baseline episodes require a pair seed and valid split")
            continue
        pairs[key].append(_variant(row))
    if any(sorted(variants) != list(range(VARIANT_COUNTS[key[0]])) for key, variants in pairs.items()):
        failures.append(
            "baseline requires unique, complete scene groups within each family and split; duplicate or partial pairs/groups found"
        )
    for index, (row, wanted) in enumerate(zip(rows, expected), start=1):
        label = f"row {index} ({row.get('name', 'unnamed')})"
        runtime_metrics = row.get("runtime_metrics", {})
        # A fixed-route pilot can exhaust a movement's time limit against the
        # closed corridor wall. This is the one accepted negative-control
        # timeout: trusted telemetry must prove the intended physical impact.
        closed_corridor_control = (
            policy == "blind_shortcut" and row.get("family") == FAMILY_THREE and _variant(row) == 1
        )
        verified_barrier_failure = (
            closed_corridor_control
            and row.get("score") == 0.0
            and _binary(runtime_metrics.get("route_barrier_hit")) == 1.0
            and _binary(runtime_metrics.get("no_collision")) == 0.0
        )
        blocked_pad_control = row.get("family") in ("alternate_pad", "go_around", "compound") and (
            (policy == "blind_shortcut" and _variant(row) != 0) or (policy == "always_b" and _variant(row) == 2)
        )
        verified_pad_failure = (
            blocked_pad_control
            and row.get("score") == 0.0
            and _binary(runtime_metrics.get("scenario_obstacle_hit")) == 1.0
            and _binary(runtime_metrics.get("no_collision")) == 0.0
        )
        if blocked_pad_control and not verified_pad_failure:
            failures.append(
                f"{label}: blocked-pad shortcut requires scenario_obstacle_hit=1 and no_collision=0 with reward 0"
            )
        for key in RUNTIME_FAILURE_METRICS:
            if key in runtime_metrics:
                value = _number(runtime_metrics[key])
                if value is None or value < 0:
                    failures.append(f"{label}: invalid runtime metric {key}")
                elif value > 0 and not (
                    key == "timed_out_actions" and (verified_barrier_failure or verified_pad_failure)
                ):
                    failures.append(f"{label}: runtime failure {key}={value:g}")
        if row.get("score") != wanted:
            failures.append(f"{label}: expected mission reward {wanted:g}, got {row.get('score')}")
        if policy == "honest" and row.get("decision_correct") != 1.0:
            failures.append(f"{label}: honest decision was not correct")
        if row.get("family") == FAMILY_THREE:
            route_choice = row.get("route_choice")
            if policy == "honest":
                if row.get("ending") != "land":
                    failures.append(f"{label}: honest obstacle_bypass must execute land")
                expected_route = "positive" if _variant(row) == 0 else "negative"
                if route_choice is not None and route_choice != expected_route:
                    failures.append(f"{label}: honest route_choice must be {expected_route}")
            elif policy == "blind_shortcut":
                if route_choice != "positive":
                    failures.append(f"{label}: obstacle_bypass blind_shortcut must choose the positive corridor")
                if closed_corridor_control and not verified_barrier_failure:
                    failures.append(
                        f"{label}: closed-corridor shortcut requires route_barrier_hit=1 and no_collision=0 with reward 0"
                    )
            elif policy == "broken" and route_choice not in (None, "none"):
                failures.append(f"{label}: broken obstacle_bypass must not choose a corridor")
        if row.get("family") in ADVANCED_FAMILIES and policy == "honest":
            expected_ending = (
                "abort"
                if (
                    (row["family"] == "alternate_pad" and _variant(row) == 2)
                    or (row["family"] == "compound" and _variant(row) == 3)
                )
                else "land"
            )
            if row.get("ending") != expected_ending:
                failures.append(f"{label}: honest {row['family']} must execute {expected_ending}")
        if policy in ("blind_shortcut", "always_b"):
            if row.get("ending") != "land":
                failures.append(f"{label}: {policy} must choose land")
            if row.get("decision_correct") != wanted:
                failures.append(f"{label}: {policy} decision does not match scene variant")
        if policy == "broken":
            if row.get("ending") != "land":
                failures.append(f"{label}: broken must execute land at launch")
            broken_decision = float(not row["blocked"]) if row["family"] == FAMILY_ONE else 0.0
            if row.get("decision_correct") != broken_decision:
                failures.append(f"{label}: broken landing decision does not match family contract")
            if "turns" in runtime_metrics and _number(runtime_metrics["turns"]) != 1.0:
                failures.append(f"{label}: broken must land on its first turn")
    return failures


def check_baseline(rows: list[dict], policy: str, expected_count: int | None = None) -> list[str]:
    """Compatibility entry point for explicit baseline acceptance."""
    return validation_errors(rows, policy=policy, expected_count=expected_count)


def summarize(rows: list[dict], policy: str | None = None) -> dict:
    """Summarize one family without counting duplicate or incomplete scene groups."""
    families = {r.get("family", "unknown") for r in rows}
    family = next(iter(families)) if len(families) == 1 else "mixed"
    pairs = defaultdict(list)
    unpaired_rows = 0
    for row in rows:
        if row.get("family") not in PAIRED_FAMILIES:
            continue
        key = _pair_key(row)
        if key is None or _variant(row) is None:
            unpaired_rows += 1
        else:
            pairs[key].append(row)
    complete, ambiguous, partial = [], 0, 0
    for key, variants in pairs.items():
        by_variant = defaultdict(list)
        for row in variants:
            by_variant[_variant(row)].append(row)
        wanted = range(VARIANT_COUNTS[key[0]])
        if any(len(group) > 1 for group in by_variant.values()):
            ambiguous += 1
        elif all(variant in by_variant for variant in wanted):
            complete.append(tuple(by_variant[variant][0] for variant in wanted))
        else:
            partial += 1

    def mean(subset):
        return round(sum(r.get("score") or 0.0 for r in subset) / len(subset), 3) if subset else None

    decisions = sum(r.get("decision_correct") == 1.0 for r in rows)
    completed = sum(r.get("completed", r.get("score") == 1.0) for r in rows)
    expected = _expected_rows(rows, policy)
    baseline_failures = check_baseline(rows, policy) if policy is not None else None
    return {
        "family": family,
        "rollouts": len(rows),
        "score": mean(rows),
        "clear_score": mean([r for r in rows if r.get("blocked") is False]),
        "blocked_score": mean([r for r in rows if r.get("blocked") is True]),
        "decisions_count": decisions,
        "decisions_rate": round(decisions / len(rows), 3) if rows else None,
        "completed_count": completed,
        "completed_rate": round(completed / len(rows), 3) if rows else None,
        "expected": round(sum(expected) / len(expected), 3) if expected else None,
        # "pairs" aliases retain compatibility with historical evidence consumers;
        # a complete group can now contain two, three or four scene variants.
        "groups": len(complete),
        "solved_groups": sum(all(row.get("decision_correct") == 1.0 for row in group) for group in complete),
        "passed_groups": sum(all(row.get("score") == 1.0 for row in group) for group in complete),
        "pairs": len(complete),
        # Both find_pad variants should end in land; ending flips are not a
        # measure of that family's change in landing location.
        "flip_rate": round(sum(a.get("ending") != b.get("ending") for a, b in complete) / len(complete), 3)
        if complete and family == FAMILY_ONE
        else None,
        "solved_pairs": sum(all(row.get("decision_correct") == 1.0 for row in group) for group in complete),
        "passed_pairs": sum(all(row.get("score") == 1.0 for row in group) for group in complete),
        "ambiguous_pairs": ambiguous,
        "partial_pairs": partial,
        "unpaired_rows": unpaired_rows,
        "errors": sum(bool(r.get("error")) for r in rows),
        "missing_metrics": sum(bool(r.get("missing_metrics")) for r in rows),
        "missing_decisions": sum("decision_correct" in r.get("missing_metrics", ()) for r in rows),
        "missing_rewards": sum("mission reward" in r.get("missing_metrics", ()) for r in rows),
        "hits": sum(bool(r.get("hit")) for r in rows),
        "baseline_ok": not baseline_failures if baseline_failures is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", metavar="LABEL=PATH")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    if args.check_baseline and args.policy is None:
        parser.error("--check-baseline requires --policy")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive")

    print(
        f"{'run':20s} {'family':18s} {'n':>4s} {'decisions':>10s} {'completed':>10s} {'expected':>8s} {'group_decisions':>15s} {'group_completed':>15s} {'errors':>6s}"
    )
    failed = False
    for arg in args.inputs:
        if "=" not in arg:
            parser.error(f"expected LABEL=PATH, got {arg!r}")
        label, path = arg.split("=", 1)
        try:
            rows = load(path)
        except (ValueError, OSError) as exc:
            print(f"{label}: ERROR: {exc}")
            failed = True
            continue
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["family"]].append(row)
        for family, family_rows in sorted(grouped.items()):
            summary = summarize(family_rows, args.policy)
            n, pairs = summary["rollouts"], summary["pairs"]
            decision_count = f"{summary['decisions_count']}/{n}" + ("?" if summary["missing_decisions"] else "")
            completed_count = f"{summary['completed_count']}/{n}" + ("?" if summary["missing_rewards"] else "")
            pair_decisions = f"{summary['solved_pairs']}/{pairs}" if pairs else "-"
            pair_completed = f"{summary['passed_pairs']}/{pairs}" if pairs else "-"
            expected = f"{summary['expected']:.3f}" if summary["expected"] is not None else "-"
            print(
                f"{label:20s} {family:18s} {n:4d} {decision_count:>10s} {completed_count:>10s} {expected:>8s} {pair_decisions:>14s} {pair_completed:>14s} {summary['errors']:6d}"
            )
            issues = {
                k: summary[k]
                for k in ("missing_metrics", "ambiguous_pairs", "partial_pairs", "unpaired_rows")
                if summary[k]
            }
            if issues:
                print(f"  {label}/{family}: " + ", ".join(f"{key}={value}" for key, value in issues.items()))
        problems = []
        if args.check_baseline:
            problems = validation_errors(rows, args.policy, args.expected_count)
        elif args.expected_count is not None:
            problems = validation_errors(rows, expected_count=args.expected_count)
        for problem in problems:
            print(f"  {label}: FAIL: {problem}")
        failed = failed or bool(problems)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
