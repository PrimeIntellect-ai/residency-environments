# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Summarize ``midog-mitosis`` eval runs as a benchmark table.

    uv run scripts/midog-mitosis/report.py ~/outputs/midog-mitosis--<model>--<harness>--<id> [...]

For each run directory (one containing ``traces.jsonl``) it prints the run as a whole (in bold) and, when both occur, the
side-by-side pairs and the single fields separately.
A row gives the pass rate (the chance one rollout passes) with a 95% interval, the pass rate of the best answer that
ignores the image, the mean F1 of the answers (how close a failed one was; it never enters the reward), the share of replies
that could be parsed, and the number of rollouts that ended in an error (left out, never averaged in silently). For an agent
harness it adds ``at turn limit``, the share of rollouts stopped by the turn cap before they answered.

With several rollouts per task it adds ``mixed``, the share of tasks some rollouts passed and others failed (the only
tasks a group-relative RL update learns from), and a second table with ``pass@k`` for k = 1 to the number of rollouts:
the chance that at least one of k rollouts of a task passes, estimated without bias from the n rollouts that were run
(Chen et al., 2021). A reward that can be learned shows pass@k rising with k; one that can be guessed would too, so read it
next to the image-blind column and the controls in the README.

The benchmark is a fixed set of tasks, so the 95% intervals measure only how much a model's results vary between
attempts: each task's attempts count as draws with that task's own pass chance (a normal approximation), and a run with
one attempt per task gets no interval. A rollout that hit an infrastructure error (sandbox, network, transient
provider fault) and then succeeded on retry counts as a normal rollout. A rollout in which any try hit the model
provider's image cap ("Downloaded image content cannot exceed 30MB") is a failed rollout with no answer: the agent's own
context reached the cap, as with a context-window overflow, so a retry that got under it does not count.

With ``--base RUN`` it also compares each run with that one on the tasks both ran (a trained policy against its base
model), as a whole and, when both occur, by layout: the pass rate of each, their difference with its interval, and the
tasks with at least one pass.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MATCH_RADIUS_PX = 30
BLIND_GRID_PX = 6
IMAGE_CAP = "Downloaded image content cannot exceed"


def slide_of(row_key: str) -> str:
    """``s0549_f03`` / ``s0549_f03+s0549_f05`` -> ``s0549``."""
    return row_key.split("_")[0]


def load(run: Path) -> tuple[dict[str, dict], int]:
    """Per-task passes, F1 and parse flags (one entry per rollout) from the finished rollouts, and the errored count."""
    tasks: dict[str, dict] = defaultdict(lambda: {"passed": [], "f1": [], "answered": [], "capped": [], "data": None})
    errors = 0
    with (run / "traces.jsonl").open() as handle:
        for line in handle:
            for trace in json.loads(line)["traces"]:
                data = trace["task"]["data"]
                rewards, metrics = trace.get("rewards") or {}, trace.get("metrics") or {}
                image_cap = any(
                    e["type"] == "ProviderError" and IMAGE_CAP in e["message"] for e in trace.get("errors") or []
                )
                if not image_cap and (not trace.get("ok") or not rewards):
                    errors += 1
                    continue
                entry = tasks[data["row_key"]]
                entry["passed"].append(
                    not image_cap and sum(r["score"] * r["weight"] for r in rewards.values()) >= 0.999
                )
                entry["f1"].append(0.0 if image_cap else metrics["find/f1"])
                entry["answered"].append(0.0 if image_cap else metrics["find/answered"])
                entry["capped"].append(trace.get("stop_condition") == "max_turns")
                entry["data"] = data
    return tasks, errors


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k rollouts passes, from c passes in n (the unbiased estimator of Chen et al., 2021)."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def attempt_variance(n: int, c: int, k: int) -> float | None:
    """Variance of one task's pass@k estimate across reruns of its n attempts (c passed); None for a single attempt.

    pass@1 uses the unbiased binomial estimate; for k > 1 it is the estimator's variance at the observed pass chance.
    """
    if n < 2:
        return None
    if k == 1:
        return c * (n - c) / (n * n * (n - 1))
    p = c / n
    chances = [math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(n + 1)]
    values = [pass_at_k(n, j, k) for j in range(n + 1)]
    mean = sum(q * v for q, v in zip(chances, values, strict=True))
    return sum(q * v * v for q, v in zip(chances, values, strict=True)) - mean * mean


def blind_pass_rate(tasks: list[dict]) -> float:
    """Share of these tasks the best fixed answer passes (chosen on these very tasks, so an upper bound).

    Only an image with a single mitotic figure can be passed without looking, by a point within the match radius of it.
    """
    hits: dict[tuple[int, int], int] = defaultdict(int)
    reach = math.ceil(MATCH_RADIUS_PX / BLIND_GRID_PX)
    for data in tasks:
        if data["targets"] != 1:
            continue
        x, y = data["points"][0]
        for gx in range(round(x / BLIND_GRID_PX) - reach, round(x / BLIND_GRID_PX) + reach + 1):
            for gy in range(round(y / BLIND_GRID_PX) - reach, round(y / BLIND_GRID_PX) + reach + 1):
                if math.dist((gx * BLIND_GRID_PX, gy * BLIND_GRID_PX), (x, y)) <= MATCH_RADIUS_PX:
                    hits[(gx, gy)] += 1
    return max(hits.values(), default=0) / len(tasks)


def interval(mean: float, variances: list[float | None]) -> tuple[float, float] | None:
    """95% interval of a mean over a fixed set of tasks, where only the attempts vary; None if a task has one attempt."""
    known = [v for v in variances if v is not None]
    if len(known) < len(variances):
        return None
    half = 1.96 * math.sqrt(sum(known)) / len(known)
    return mean - half, mean + half


def row(label: str, entries: list[dict], errors: int, grouped: bool, bold: bool, capped: bool = False) -> list[str]:
    rate = statistics.fmean(statistics.fmean(e["passed"]) for e in entries)
    bounds = interval(rate, [attempt_variance(len(e["passed"]), sum(e["passed"]), 1) for e in entries])
    cells = [
        f"**{label}**" if bold else label,
        str(len(entries)),
        str(len({slide_of(e["data"]["row_key"]) for e in entries})),
        f"**{rate:.3f}**" if bold else f"{rate:.3f}",
        "—" if bounds is None else f"{max(bounds[0], 0.0):.3f} to {min(bounds[1], 1.0):.3f}",
        f"{blind_pass_rate([e['data'] for e in entries]):.3f}",
        f"{statistics.fmean(f for e in entries for f in e['f1']):.3f}",
    ]
    if grouped:
        cells.append(f"{statistics.fmean(0 < sum(e['passed']) < len(e['passed']) for e in entries):.0%}")
    if capped:
        cells.append(f"{statistics.fmean(c for e in entries for c in e['capped']):.0%}")
    cells += [f"{statistics.fmean(a for e in entries for a in e['answered']):.0%}", str(errors)]
    return cells


def curve_row(label: str, entries: list[dict], most: int, bold: bool) -> list[str]:
    """pass@k for k = 1..most over the tasks with at least k rollouts; "—" when none has that many."""
    cells = [f"**{label}**" if bold else label]
    for k in range(1, most + 1):
        usable = [e for e in entries if len(e["passed"]) >= k]
        if not usable:
            cells.append("—")
            continue
        value = statistics.fmean(pass_at_k(len(e["passed"]), sum(e["passed"]), k) for e in usable)
        bounds = interval(value, [attempt_variance(len(e["passed"]), sum(e["passed"]), k) for e in usable])
        notes = [] if bounds is None else [f"{max(bounds[0], 0.0):.1%} to {min(bounds[1], 1.0):.1%}"]
        notes += [f"{len(usable)} tasks"] if len(usable) < len(entries) else []
        cells.append(f"{value:.1%}" + (f" ({', '.join(notes)})" if notes else ""))
    return cells


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"|{' --- |' * len(header)}"]
    return "\n".join(lines + [f"| {' | '.join(cells)} |" for cells in rows])


def paired(base: dict[str, dict], run: dict[str, dict]) -> str:
    """Pass rate of both runs on the tasks both ran, their difference with its interval, and the tasks with a pass."""
    common = [k for k in run if k in base]
    layouts = [("pairs", [k for k in common if "+" in k]), ("fields", [k for k in common if "+" not in k])]
    rows = []
    for label, keys in [("all", common)] + (layouts if all(group for _, group in layouts) else []):
        if not keys:
            continue
        rate = {
            name: {k: statistics.fmean(runs[k]["passed"]) for k in keys}
            for name, runs in (("base", base), ("run", run))
        }
        mean = statistics.fmean(rate["run"][k] - rate["base"][k] for k in keys)
        variances: list[float | None] = []
        for k in keys:
            parts = [attempt_variance(len(runs[k]["passed"]), sum(runs[k]["passed"]), 1) for runs in (base, run)]
            variances.append(None if None in parts else sum(parts))
        bounds = interval(mean, variances)
        with_pass = [sum(any(runs[k]["passed"]) for k in keys) for runs in (base, run)]
        rows.append(
            [
                label,
                str(len(keys)),
                f"{statistics.fmean(rate['base'].values()):.3f}",
                f"{statistics.fmean(rate['run'].values()):.3f}",
                f"{mean:+.3f}",
                "—" if bounds is None else f"{bounds[0]:+.3f} to {bounds[1]:+.3f}",
                f"{with_pass[0]} to {with_pass[1]}",
            ]
        )
    return table(["tasks", "common tasks", "base", "run", "difference", "95% CI", "tasks with a pass"], rows)


def report(run: Path) -> str:
    tasks, errors = load(run)
    entries = list(tasks.values())
    most = max(len(e["passed"]) for e in entries)
    grouped = most > 1
    layouts = [
        ("pairs", [e for e in entries if "+" in e["data"]["row_key"]], False),
        ("fields", [e for e in entries if "+" not in e["data"]["row_key"]], False),
    ]
    tiers = [("all", entries, True)] + ([t for t in layouts if t[1]] if all(group for _, group, _ in layouts) else [])
    capped = any(c for e in entries for c in e["capped"])
    header = ["tasks", "n", "slides", "pass rate", "95% CI", "image-blind", "mean F1"]
    header += ["mixed"] if grouped else []
    header += ["at turn limit"] if capped else []
    header += ["parsed", "errors"]
    out = [table(header, [row(label, group, errors, grouped, bold, capped) for label, group, bold in tiers])]
    if grouped:
        out.append("\npass@k, the chance that at least one of k rollouts of a task passes:\n")
        out.append(
            table(
                ["tasks"] + [f"pass@{k}" for k in range(1, most + 1)], [curve_row(l, g, most, b) for l, g, b in tiers]
            )
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing traces.jsonl.")
    parser.add_argument("--base", type=Path, help="Also compare each run with this one, task by task.")
    args = parser.parse_args()
    base = load(args.base.expanduser())[0] if args.base else None
    for run in args.runs:
        text = report(run.expanduser())
        if base is not None:
            text += "\n\npaired against the base run:\n\n" + paired(base, load(run.expanduser())[0])
        print(f"### {run.name}\n\n{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
