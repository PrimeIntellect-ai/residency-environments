"""What do a run's answers look like, and are its passes specific to the image?

    uv run --no-sync python scripts/midog-mitosis/specificity.py ~/outputs/<run-dir> [...]

Needs the package installed (``uv pip install -e ./environments/midog-mitosis``). Two tables per run.

*Counts.* Per tier, how many points the answers list against the number of figures: fewer (the model stops early), the right
number, or more (it pads); how many of the right-count answers pass, which isolates localization from enumeration; and how many
answers contain every figure among their points, extras or not, which is what a model that could prune its list would pass.

*Specificity.* Every parsed answer is scored three ways. ``own`` is its pass rate on its own task. ``other tasks`` scores the
same answer against every other task whose image has the same size, per pair. ``random`` places the same number of points
uniformly at random in its own image (100 draws, 40 px from the border, as the figures are). An answer that passes because it
looks at the image passes its own task far more often than the other two; a generic coordinate would not. One random point
passes a one-figure image about 0.4% of the time, so ``other tasks`` and ``random`` are not zero for one-point answers, and a
model should be read against them rather than against zero.

*Prior.* A single point near the middle of the image passes a few percent of the one-figure images (``report.py``'s
image-blind column). A passing one-point answer is *generic* when it would also pass at least 2% of the other one-figure
images of the run, i.e. when it sits on that prior instead of on this image's figure; the line below the tables counts them.
"""

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import verifiers.v1 as vf
from midog_mitosis.scoring import MATCH_RADIUS_PX, POINT_SCALE, max_matching, parse_points
from midog_mitosis.taskset import final_reply

DRAWS = 100
BORDER_PX = 40
GENERIC_RATE = 0.02
GENERIC_MIN_IMAGES = 3


def load(run: Path) -> tuple[dict[str, dict], list[tuple[str, list | None, bool]]]:
    tasks, rollouts = {}, []
    for line in (run / "traces.jsonl").read_text().splitlines():
        for trace in json.loads(line)["traces"]:
            data = trace["task"]["data"]
            tasks[data["row_key"]] = data
            if not trace.get("ok") or not trace.get("rewards"):
                continue
            text = final_reply(vf.Trace.model_validate(copy.deepcopy(trace)))
            passed = sum(r["score"] * r["weight"] for r in trace["rewards"].values()) >= 0.999
            rollouts.append((data["row_key"], parse_points(text), passed))
    return tasks, rollouts


def exact(points_px: list[tuple[float, float]], data: dict) -> bool:
    gold = [tuple(p) for p in data["points"]]
    return len(points_px) == len(gold) and max_matching(points_px, gold, MATCH_RADIUS_PX) == len(gold)


def pixels(points: list[tuple[float, float]], data: dict) -> list[tuple[float, float]]:
    return [(x * data["width"] / POINT_SCALE, y * data["height"] / POINT_SCALE) for x, y in points]


def counts(tasks: dict, rollouts: list) -> str:
    rows = [
        "| tasks | rollouts | fewer points than figures | the right number | more | unparsed | right-count answers that pass | contain every figure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tier, keep in (("pairs", lambda d: "+" in d["row_key"]), ("fields", lambda d: "+" not in d["row_key"])):
        group = [(tasks[key], points, passed) for key, points, passed in rollouts if keep(tasks[key])]
        if not group:
            continue
        size = Counter()
        right_pass = []
        for data, points, passed in group:
            if points is None:
                size["unparsed"] += 1
                continue
            gold = [tuple(p) for p in data["points"]]
            size["complete"] += max_matching(pixels(points, data), gold, MATCH_RADIUS_PX) == len(gold)
            if len(points) < data["targets"]:
                size["fewer"] += 1
            elif len(points) == data["targets"]:
                size["right"] += 1
                right_pass.append(passed)
            else:
                size["more"] += 1
        n = len(group)
        rate = f"{sum(right_pass) / len(right_pass):.0%}" if right_pass else "n/a"
        rows.append(
            f"| {tier} | {n} | {size['fewer'] / n:.0%} | {size['right'] / n:.0%} | {size['more'] / n:.0%} | "
            f"{size['unparsed'] / n:.0%} | {rate} | {size['complete'] / n:.0%} |"
        )
    return "\n".join(rows)


def specificity(tasks: dict, rollouts: list) -> str:
    rng = random.Random(0)
    by_size = defaultdict(list)
    for key, data in tasks.items():
        by_size[(data["width"], data["height"])].append(key)
    # own passes, answers, cross passes, pairs, random passes, draws
    totals: dict[str, list[int]] = defaultdict(lambda: [0] * 6)
    for key, points, passed in rollouts:
        if not points:
            continue
        data = tasks[key]
        answer_px = pixels(points, data)
        row = totals["1 point" if len(points) == 1 else "2+ points"]
        row[0] += passed
        row[1] += 1
        for other in by_size[(data["width"], data["height"])]:
            if other != key:
                row[2] += exact(answer_px, tasks[other])
                row[3] += 1
        for _ in range(DRAWS):
            guess = [
                (rng.uniform(BORDER_PX, data["width"] - BORDER_PX), rng.uniform(BORDER_PX, data["height"] - BORDER_PX))
                for _ in points
            ]
            row[4] += exact(guess, data)
            row[5] += 1
    totals["all parsed"] = [sum(row[i] for key, row in totals.items()) for i in range(6)]
    lines = [
        "| answers | n | own task | same answers on other tasks (per pair) | same number of random points |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("1 point", "2+ points", "all parsed"):
        own, n, cross, pairs, rand, draws = totals[label]
        if n:
            lines.append(
                f"| {label} | {n} | {own / n:.2%} | {cross / max(pairs, 1):.3%} | {rand / max(draws, 1):.3%} |"
            )
    return "\n".join(lines)


def reference_golds(tasks: dict, dataset: str | None, split: str) -> list[tuple[float, float]]:
    """Gold centers (pixels) of the one-figure 768 px images that define the prior: a split of the dataset's ``warmup`` config, or the run's own tasks."""
    if dataset is None:
        return [tuple(d["points"][0]) for d in tasks.values() if d["targets"] == 1 and d["width"] == d["height"]]
    from datasets import Dataset, load_dataset

    root = Path(dataset).expanduser()
    if root.is_dir():
        rows = Dataset.from_parquet(
            sorted(str(path) for path in (root.resolve() / "warmup").glob(f"{split}-*.parquet"))
        )
    else:
        rows = load_dataset(dataset, "warmup", split=split)
    rows = rows.select_columns(["n_targets", "width", "height", "points"])
    return [tuple(r["points"][0]) for r in rows if r["n_targets"] == 1 and r["width"] == r["height"]]


def prior(tasks: dict, rollouts: list, reference: list[tuple[float, float]]) -> str:
    if not reference:
        return "No one-figure tasks in this run, so no single-point pass can sit on the central-point prior."
    passes = generic = 0
    for key, points, passed in rollouts:
        data = tasks[key]
        if passed and points and len(points) == 1 and data["targets"] == 1 and data["width"] == data["height"]:
            x, y = pixels(points, data)[0]
            own = tuple(data["points"][0])
            others = [gold for gold in reference if math.dist(gold, own) > 0.5]
            reach = sum(math.dist((x, y), gold) <= MATCH_RADIUS_PX for gold in others)
            passes += 1
            generic += reach >= max(GENERIC_MIN_IMAGES, GENERIC_RATE * len(others))
    return (
        f"One-figure passes by a single point: {passes}; generic (would also pass at least {GENERIC_RATE:.0%} of the other "
        f"{len(reference) - 1} one-figure images, and at least {GENERIC_MIN_IMAGES}): {generic}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing traces.jsonl.")
    parser.add_argument(
        "--reference",
        help="Dataset (repo id or local build directory) whose one-figure tasks define the prior; default: the run's own tasks "
        "(too few for a logged training sample, which covers a fraction of the split).",
    )
    parser.add_argument("--reference-split", default="test")
    args = parser.parse_args()
    for run in args.runs:
        tasks, rollouts = load(run.expanduser())
        reference = reference_golds(tasks, args.reference, args.reference_split)
        print(
            f"### {run.name}\n\n{counts(tasks, rollouts)}\n\n{specificity(tasks, rollouts)}\n\n"
            f"{prior(tasks, rollouts, reference)}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
