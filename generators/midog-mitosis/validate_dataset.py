# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=4.0",
#     "imagecodecs",
#     "numpy>=2.1",
#     "pandas>=2.2",
#     "pillow>=11.0",
#     "pyarrow>=17.0",
#     "tifffile",
# ]
# ///
"""Validate a built ``midog-mitosis`` dataset against the MIDOG++ source data.

Independent of the builder's image and sampling code. From the released ``MIDOG++.json`` and
``datasets_xvalidation.csv`` (cached by the build under ``<work-dir>/source``) it recomputes every stored
label, vote string, coordinate and tumor type, and verifies the recorded custom slide partition across every config.
``--spot-check N`` additionally re-downloads N slides, verifies their MD5 and
re-crops every stored field and candidate at its recorded coordinates, so a wrong slice, transposed axes or an
off-by-one origin cannot pass.

It then checks the ``tasks`` config (at least three figures per image) and the ``warmup`` config (one figure per field)
against those verified source tables: every task's expected answer and points are recomputed from its fields, each
clean by the experts' votes. Images must match their source fields, and no benchmark field or image may be reused.
Full builds must contain 512 test tasks and 64 validation tasks, preserving all 36 retained tasks and their image hashes.
Warmups cover the clean one-figure fields that no benchmark task shows; training tasks cover the eligible three-or-more-figure fields.
The environment's scorer accepts the expected answers and rejects invalid ones, and the answers an adversary could
give without looking pass none of the benchmark (random points) and at most a few percent of the warm-up images (one fixed
point fitted on the train images).

    uv run generators/midog-mitosis/validate_dataset.py --work-dir ~/midog-build

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

EDGE = 40
FIELD_SIZE = 768
CANDIDATE_NATIVE = 128
CANDIDATE_OUT = 512
BOX_SIZE = 50
MAX_CELLS = 12
SPLITS = ("train", "validation", "test")
CLASS_NAMES = {1: "mitotic figure", 2: "imposter"}
TUMORS = {
    "human breast cancer": ("breast carcinoma", "human"),
    "canine lung cancer": ("lung carcinoma", "canine"),
    "canine lymphoma": ("lymphoma", "canine"),
    "canine lymphosarcoma": ("lymphoma", "canine"),
    "canine cutaneous mast cell tumor": ("mast cell tumor", "canine"),
    "human neuroendocrine tumor": ("neuroendocrine tumor", "human"),
    "canine soft tissue sarcoma": ("soft tissue sarcoma", "canine"),
    "human melanoma": ("melanoma", "human"),
}
DOWNLOAD = "https://ndownloader.figshare.com/files/{id}"
USER_AGENT = "midog-mitosis-validate/1.0 (open-source dataset validation; github.com/dylantirandaz)"
MAD_MAX = 6.0
"""Largest mean absolute pixel difference (0-255) between a stored JPEG and an independent re-crop. JPEG quality 92
costs about 1-2 gray levels; a misplaced crop of tissue costs far more."""
MAD_SHIFT_GAP = 3.0
"""A crop shifted by 8 px must be at least this much worse than the recorded position, so the check can discriminate."""


def load_module(name: str):
    """Load ``environments/midog-mitosis/midog_mitosis/<name>.py``: ``scoring`` imports only the standard library."""
    path = Path(__file__).resolve().parents[2] / "environments" / "midog-mitosis" / "midog_mitosis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"midog_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"midog_{name}"] = module
    spec.loader.exec_module(module)
    return module


BLIND_GRID = 6
"""Step (px) of the grid of fixed points an answer that ignores the image could pick from."""
BLIND_MAX = 0.06
"""Largest share of the one-figure test images a fixed point chosen on the train images may pass."""
BLIND_RANDOM_MAX = 0.002
"""Largest share of random answers with the true count that may pass the benchmark."""


def best_fixed_point(golds: list[tuple[float, float]], radius: float) -> tuple[tuple[int, int], int]:
    """The grid point within ``radius`` of the most golds, and how many that is."""
    hits: Counter = Counter()
    reach = math.ceil(radius / BLIND_GRID)
    for x, y in golds:
        for gx in range(round(x / BLIND_GRID) - reach, round(x / BLIND_GRID) + reach + 1):
            for gy in range(round(y / BLIND_GRID) - reach, round(y / BLIND_GRID) + reach + 1):
                if math.dist((gx * BLIND_GRID, gy * BLIND_GRID), (x, y)) <= radius:
                    hits[(gx * BLIND_GRID, gy * BLIND_GRID)] += 1
    return hits.most_common(1)[0] if hits else ((0, 0), 0)


def check_tasks(
    root: Path,
    report: dict,
    raw: dict,
    tables,
    images,
    scoring,
    check,
    config: str,
    seen: set[str],
    benchmark: dict[str, set[str]],
) -> None:
    """Check every task of one config against the (already verified) source tables it points to.

    ``tasks`` holds three or more figures; ``warmup`` holds one-figure fields that no ``tasks`` row shows. ``benchmark``
    collects the fields of each split's ``tasks`` rows, so the ``tasks`` config must be checked first."""
    from datasets import Dataset
    from datasets import Image as HFImage

    def ok(condition: bool, message: str) -> bool:
        check(condition, message)
        return bool(condition)

    votes = {}
    for a in raw["annotations"]:
        x0, y0, x1, y1 = a["bbox"]
        votes[(a["image_id"], (x0 + x1) / 2, (y0 + y1) / 2)] = tuple(a["labels"])

    def clean(field: dict) -> bool:
        def cells(points):
            return [votes[(field["slide"], field["x0"] + x, field["y0"] + y)] for x, y in points]

        return all(v == (1, 1) for v in cells(field["mitoses"])) and all(
            v in {(2, 2), (2, 2, 2)} for v in cells(field["imposters"])
        )

    settings = report.get("task_settings", {})
    for key, value in {"match_radius_px": scoring.MATCH_RADIUS_PX, "point_scale": scoring.POINT_SCALE}.items():
        ok(
            settings.get(key) == value,
            f"build_report task_settings[{key!r}] is {settings.get(key)!r}, expected {value!r}",
        )
    radius = scoring.MATCH_RADIUS_PX
    rng = np.random.default_rng(0)
    train_golds: list[tuple[float, float]] = []
    print(f"\n{config} (answers, points and images checked against source fields; blind attacks):")
    for split in SPLITS:
        dataset = Dataset.from_parquet(sorted(str(path) for path in (root / config).glob(f"{split}-*.parquet")))
        rows = dataset.select_columns([c for c in dataset.column_names if c != "image"]).to_list()
        pictures = dataset.select_columns(["image"]).cast_column("image", HFImage(decode=False))
        listed = report["configs"][config][split]
        if config == "tasks" and not report["partial"] and split in ("test", "validation"):
            expected_count = {"test": 512, "validation": 64}[split]
            ok(len(rows) == expected_count, f"{config}/{split}: expected {expected_count} tasks, got {len(rows)}")
        ok(len(rows) == listed["rows"], f"{config}/{split}: {len(rows)} rows, report says {listed['rows']}")
        by_targets = {str(k): v for k, v in sorted(Counter(min(t["n_targets"], 4) for t in rows).items())}
        ok(by_targets == listed["by_targets"], f"{config}/{split}: target counts {by_targets} differ from the report")
        ok(
            sum(len(t["parts"]) == 2 for t in rows) == listed["pairs"],
            f"{config}/{split}: pair count differs from the report",
        )
        fields = tables[("fields", split)]
        where = {r["field_id"]: i for i, r in enumerate(fields)}
        in_pair: Counter = Counter()
        natural: set[str] = set()
        used: Counter = Counter()
        picture_hashes: set[str] = set()
        retained = (
            {task["task_id"]: task for task in report["provenance"]["retained_tasks"]}
            if config == "tasks" and split == "test"
            else {}
        )
        if config == "tasks" and split == "test" and not report["partial"]:
            ok(len(retained) == 36, "test must retain 36 original hard tasks")
            ok(set(retained) <= {task["task_id"] for task in rows}, "test is missing retained hard tasks")
        for i, t in enumerate(rows):
            tid, parts = t["task_id"], t["parts"]
            if not ok(
                len(parts) in ((1, 2) if config == "tasks" else (1,)) and all(p in where for p in parts),
                f"{tid}: parts {parts} are not fields of {split}",
            ):
                continue
            srcs = [fields[where[p]] for p in parts]
            ok(t["source_id"] == "+".join(parts), f"{tid}: source_id does not join the parts")
            ok(tid == f"find:{t['source_id']}", f"{tid}: task_id does not match source_id")
            ok(tid not in seen, f"{tid}: duplicate task_id")
            seen.add(tid)
            used.update(parts)
            for src in srcs:
                ok(clean(src), f"{tid}: field {src['field_id']} holds a disputed cell")
                ok(src["slide"] == srcs[0]["slide"], f"{tid}: parts come from different slides")
            ok(any(src["mitoses"] for src in srcs), f"{tid}: no mitotic figure in any part")
            for key in ("slide", "tumor_type", "species", "scanner", "origin"):
                ok(t[key] == srcs[0][key], f"{tid}: {key} {t[key]!r} differs from the source row {srcs[0][key]!r}")
            size = srcs[0]["size"]
            width = size * len(parts)
            ok(
                (t["width"], t["height"]) == (width, size),
                f"{tid}: size {t['width']}x{t['height']}, expected {width}x{size}",
            )
            picture = pictures[i]["image"]["bytes"]
            digest = hashlib.sha256(picture).hexdigest()
            ok(digest not in picture_hashes, f"{tid}: repeated image in {config}/{split}")
            picture_hashes.add(digest)
            if tid in retained:
                original = retained[tid]
                ok(parts == original["parts"], f"{tid}: retained parts changed")
                ok(digest == original["image_sha256"], f"{tid}: retained image changed")
            ok(Image.open(io.BytesIO(picture)).size == (width, size), f"{tid}: image is not {width}x{size}")
            if len(parts) == 1:
                natural.add(parts[0])
                ok(
                    picture == images[("fields", split)][where[parts[0]]]["image"]["bytes"],
                    f"{tid}: image differs from the source image",
                )
            else:
                ok(parts[0] != parts[1], f"{tid}: a pair of one field")
                ok(
                    any(len(s["mitoses"]) == 1 for s in srcs),
                    f"{tid}: a pair without a field with exactly one mitotic figure",
                )
                for p in parts:
                    in_pair[p] += 1
                whole = np.asarray(Image.open(io.BytesIO(picture)).convert("RGB"), dtype=np.int16)
                halves = [whole[:, k * size : (k + 1) * size] for k in range(2)]
                originals = [
                    np.asarray(
                        Image.open(io.BytesIO(images[("fields", split)][where[p]]["image"]["bytes"])).convert("RGB"),
                        dtype=np.int16,
                    )
                    for p in parts
                ]
                for k in range(2):
                    ok(
                        mad(halves[k], originals[k]) <= MAD_MAX,
                        f"{tid}: half {k} differs from field {parts[k]} (MAD {mad(halves[k], originals[k]):.1f})",
                    )
                    ok(mad(halves[k], originals[1 - k]) > MAD_MAX, f"{tid}: half {k} also matches the other field")
            expected = sorted((x + size * k, y) for k, s in enumerate(srcs) for x, y in s["mitoses"])
            points = [tuple(p) for p in t["points"]]
            ok(
                len(points) == len(expected) and all(math.dist(p, q) < 1e-3 for p, q in zip(points, expected)),
                f"{tid}: points differ from the source mitoses",
            )
            ok(t["n_targets"] == len(expected) >= 1, f"{tid}: n_targets {t['n_targets']}, expected {len(expected)}")
            ok(
                t["n_targets"] >= 3 if config == "tasks" else len(parts) == 1 and t["n_targets"] == 1,
                f"{tid}: {config} holds a {'pair' if len(parts) == 2 else 'field'} with {t['n_targets']} figures",
            )
            ok(
                t["n_imposters"] == sum(len(s["imposters"]) for s in srcs),
                f"{tid}: n_imposters differs from the source fields",
            )
            scale_x, scale_y = scoring.POINT_SCALE / width, scoring.POINT_SCALE / size
            want = "[" + ", ".join(f"[{x * scale_x:.1f}, {y * scale_y:.1f}]" for x, y in expected) + "]"
            ok(t["answer"] == want, f"{tid}: answer {t['answer']!r}, expected {want!r}")

            # the environment's scorer: the expected answer earns full marks, any deviation none
            def score(pts: list[tuple[float, float]], width=width, size=size) -> float:
                text = "Answer: [" + ", ".join(f"[{x * scale_x:.3f}, {y * scale_y:.3f}]" for x, y in pts) + "]"
                return scoring.score_find(text, expected, width, size).reward

            far = next(
                (
                    c
                    for c in ((100, 100), (width - 100, size - 100), (width / 2, size / 2), (100, size - 100))
                    if min(math.dist(c, g) for g in expected) > 2 * radius
                ),
                (5.0, 5.0),  # fields keep every figure at least 40 px from the border, so this corner is out of reach
            )
            ok(
                scoring.score_find(f"Answer: {t['answer']}", expected, width, size).reward == 1.0,
                f"{tid}: the scorer rejects the expected answer",
            )
            ok(score([(x + 20, y + 15) for x, y in expected]) == 1.0, f"{tid}: the scorer rejects points 25 px off")
            ok(
                score([far] + expected[1:]) == 0.0,
                f"{tid}: the scorer accepts a point far from every figure",
            )
            ok(score(expected + [far]) == 0.0, f"{tid}: the scorer accepts an extra point")
            ok(score(expected + expected[:1]) == 0.0, f"{tid}: the scorer accepts a duplicated point")
            ok(score(expected[1:]) == 0.0, f"{tid}: the scorer accepts a missing point")
            ok(
                scoring.score_find("Answer: none", expected, width, size).reward == 0.0,
                f"{tid}: the scorer accepts an unreadable answer",
            )
        clean_fields = {f["field_id"] for f in fields if clean(f)}
        figures = {f["field_id"]: len(f["mitoses"]) for f in fields}
        if config == "tasks":
            benchmark[split] = set(used)
            wanted = {f for f in clean_fields if figures[f] >= 3 and f not in in_pair}
            ok(natural <= wanted, f"{config}/{split}: an ineligible one-field task")
            ok(all(n == 1 for n in used.values()), f"{config}/{split}: a benchmark field is reused")
            if split == "train" or report["partial"]:
                ok(natural == wanted, f"{config}/{split}: missing eligible training fields")
        else:
            shown = natural & benchmark[split]
            ok(not shown, f"{config}/{split}: {len(shown)} warmup fields are also shown in benchmark tasks")
            wanted = {f for f in clean_fields if figures[f] == 1} - benchmark[split]
            ok(
                natural == wanted,
                f"{config}/{split}: warmups do not cover the undisputed one-figure fields outside benchmark tasks",
            )
        ok(sorted(t["rank"] for t in rows) == list(range(len(rows))), f"{config}/{split}: ranks are not 0..n-1")
        # what an answer that ignores the image can pass
        gold_of = lambda t: [tuple(p) for p in t["points"]]  # noqa: E731
        pairs = sum(len(t["parts"]) == 2 for t in rows)
        if config == "tasks":
            passed = 0
            for t in rows:
                for _ in range(200):
                    guess = rng.uniform(0, 1, size=(t["n_targets"], 2)) * np.array([t["width"], t["height"]])
                    passed += scoring.max_matching([tuple(g) for g in guess], gold_of(t), radius) == t["n_targets"]
            ok(
                passed <= BLIND_RANDOM_MAX * 200 * len(rows),
                f"{config}/{split}: {passed} of {200 * len(rows)} random answers with the true count passed",
            )
            print(
                f"  {split:10s} tasks {len(rows):4d}  by figures {by_targets}  pairs {pairs:3d}"
                f"  {200 * len(rows)} random answers: {passed} passed"
            )
        else:
            singles = [gold_of(t)[0] for t in rows if t["width"] == t["height"]]
            if split == "train":
                train_golds = singles
                point, hits = best_fixed_point(singles, radius)
                share = hits / max(len(singles), 1)
            else:
                point, _ = best_fixed_point(train_golds, radius)
                share = sum(math.dist(point, g) <= radius for g in singles) / max(len(singles), 1)
                ok(
                    share <= BLIND_MAX,
                    f"{config}/{split}: one fixed point fitted on train passes {share:.1%} of the one-figure images",
                )
            print(
                f"  {split:10s} tasks {len(rows):4d}  fixed point {point} passes {share:5.2%} of {len(singles)} one-figure images"
                f"{' (fitted on these)' if split == 'train' else ' (fitted on train)'}"
            )


def fetch(url: str, dest: Path, md5: str) -> None:
    for attempt in range(4):
        digest = hashlib.md5()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
        except OSError:
            time.sleep(10 * (attempt + 1))
            continue
        if digest.hexdigest() == md5:
            return
    raise RuntimeError(f"could not download {dest.name} with a matching MD5")


def rank_auc(positive: list[float], negative: list[float]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    ranks = pd.Series(positive + negative).rank()
    return float(
        (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def pick_spot_check_slides(slides: dict[str, set[int]], count: int) -> list[int]:
    """Evenly spaced slides, round-robin over splits so every split is covered."""
    queues = {split: sorted(s) for split, s in slides.items() if s}
    chosen: list[int] = []
    for round_ in range(count):
        for split, queue in queues.items():
            if len(chosen) < count and queue:
                chosen.append(queue.pop(len(queue) * round_ // max(count, 1) % len(queue)))
    return sorted(chosen)


def spot_check(work: Path, tables, images, wanted: list[int], check) -> None:
    manifest = json.loads((work / "manifest.json").read_text())
    for slide in wanted:
        entry = manifest[f"{slide:03d}.tiff"]
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            path = Path(tmp) / f"{slide:03d}.tiff"
            fetch(DOWNLOAD.format(id=entry["id"]), path, entry["md5"])
            pixels = tifffile.imread(path)[..., :3]
        height, width = pixels.shape[:2]
        fields = candidates = 0
        worst = 0.0
        for (config, split), rows in tables.items():
            for i, row in enumerate(rows):
                if row["slide"] != slide:
                    continue
                stored = np.asarray(Image.open(io.BytesIO(images[(config, split)][i]["image"]["bytes"])).convert("RGB"))
                if config == "fields":
                    x0, y0, size = row["x0"], row["y0"], row["size"]
                    truth = pixels[y0 : y0 + size, x0 : x0 + size]
                    shift = 8 if x0 + size + 8 <= width else -8
                    moved = pixels[y0 : y0 + size, x0 + shift : x0 + shift + size]
                    fields += 1
                else:
                    x, y, half = row["x"], row["y"], CANDIDATE_NATIVE // 2
                    crop = pixels[y - half : y + half, x - half : x + half]
                    truth = np.asarray(Image.fromarray(crop).resize((CANDIDATE_OUT,) * 2, Image.BICUBIC))
                    shift = 8 if x + half + 8 <= width else -8
                    moved_crop = pixels[y - half : y + half, x + shift - half : x + shift + half]
                    moved = np.asarray(Image.fromarray(moved_crop).resize((CANDIDATE_OUT,) * 2, Image.BICUBIC))
                    candidates += 1
                ident = row.get("field_id") or row["candidate_id"]
                check(truth.shape == stored.shape, f"slide {slide} {ident}: shape {truth.shape} vs {stored.shape}")
                if truth.shape != stored.shape:
                    continue
                here, there = mad(truth, stored), mad(moved, stored)
                worst = max(worst, here)
                check(here < MAD_MAX, f"slide {slide} {ident}: pixels differ from re-crop (MAD {here:.1f})")
                check(
                    there > here + MAD_SHIFT_GAP,
                    f"slide {slide} {ident}: re-crop shifted 8 px matches as well (MAD {here:.1f} vs {there:.1f})",
                )
        print(
            f"  spot-check slide {slide}: {fields} fields, {candidates} candidates, worst MAD {worst:.2f}", flush=True
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default="~/midog-build")
    parser.add_argument("--dataset", help="Dataset directory (default: <work-dir>/dataset).")
    parser.add_argument("--spot-check", type=int, default=6, help="Slides to re-download and re-crop (0 to skip).")
    parser.add_argument(
        "--partial", action="store_true", help="Dataset covers only some slides (skip completeness checks)."
    )
    args = parser.parse_args()
    work = Path(args.work_dir).expanduser()
    root = Path(args.dataset).expanduser() if args.dataset else work / "dataset"

    from datasets import Dataset
    from datasets import Image as HFImage

    scoring = load_module("scoring")
    raw = json.loads((work / "source" / "MIDOG++.json").read_text())
    official = pd.read_csv(work / "source" / "datasets_xvalidation.csv", sep=";").set_index("Slide")
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    sizes = {i["id"]: (i["width"], i["height"]) for i in raw["images"]}
    annotations: dict[int, dict[int, dict]] = defaultdict(dict)
    for a in raw["annotations"]:
        x0, y0, x1, y1 = a["bbox"]
        check(x1 - x0 == BOX_SIZE and y1 - y0 == BOX_SIZE, f"annotation {a['id']}: box is {x1 - x0}x{y1 - y0}")
        annotations[a["image_id"]][a["id"]] = {
            "cls": a["category_id"],
            "x": (x0 + x1) / 2,
            "y": (y0 + y1) / 2,
            "votes": a["labels"],
        }
    report = json.loads((root / "build_report.json").read_text())
    partial = args.partial or bool(report.get("partial"))

    tables: dict[tuple[str, str], list[dict]] = {}
    images: dict[tuple[str, str], object] = {}
    for config in ("fields", "candidates"):
        for split in SPLITS:
            dataset = Dataset.from_parquet(sorted(str(path) for path in (root / config).glob(f"{split}-*.parquet")))
            images[(config, split)] = dataset.select_columns(["image"]).cast_column("image", HFImage(decode=False))
            tables[(config, split)] = dataset.select_columns(
                [c for c in dataset.column_names if c != "image"]
            ).to_list()
            expected = report["configs"][config][split]["rows"]
            check(
                len(tables[(config, split)]) == expected,
                f"{config}/{split}: {len(tables[(config, split)])} rows, report says {expected}",
            )
        id_key = "field_id" if config == "fields" else "candidate_id"
        ids = [r[id_key] for split in SPLITS for r in tables[(config, split)]]
        check(len(ids) == len(set(ids)), f"{config}: duplicate {id_key}")

    # Every source row, including warmups' source fields, follows the same whole-slide partition.
    slides = {split: {r["slide"] for c in ("fields", "candidates") for r in tables[(c, split)]} for split in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                check(
                    not slides[a] & slides[b], f"slides shared between {a} and {b}: {sorted(slides[a] & slides[b])[:5]}"
                )
    declared = {split: set(report["provenance"]["split_assignments"][split]) for split in SPLITS}
    official_test = {int(s) for s in official.index[official.Dataset == "test"]}
    check(official_test <= declared["test"], "the expanded test set drops an official test slide")
    check(set.union(*declared.values()) == set(official.index), "split manifest does not cover all 503 source slides")
    for a in SPLITS:
        check(slides[a] <= declared[a], f"{a}: source rows violate the slide manifest")
        for b in SPLITS:
            if a < b:
                check(not declared[a] & declared[b], f"split manifest shares slides between {a} and {b}")
    if not partial:
        for split in SPLITS:
            check(slides[split] == declared[split], f"{split}: source rows do not cover the declared slides")
        for split in SPLITS:
            check(
                {r["slide"] for r in tables[("fields", split)]} == slides[split],
                f"fields/{split} does not cover every {split} slide",
            )
    print({split: len(s) for split, s in slides.items()}, "slides", "(partial)" if partial else "")

    # candidates: from every slide, the same number of each class: min(cap, usable mitotic figures, usable imposters)
    half, cap = CANDIDATE_NATIVE // 2, report["settings"]["candidates_per_class"]
    for split in SPLITS:
        got = Counter((r["slide"], r["label"]) for r in tables[("candidates", split)])
        for slide in slides[split]:
            width, height = sizes[slide]
            usable = Counter(
                CLASS_NAMES[a["cls"]]
                for a in annotations[slide].values()
                if half <= a["x"] < width - half and half <= a["y"] < height - half
            )
            want = min(cap, usable["mitotic figure"], usable["imposter"])
            for label in CLASS_NAMES.values():
                check(
                    got[(slide, label)] == want, f"slide {slide}: {got[(slide, label)]} {label} candidates, want {want}"
                )

    # every image decodes at the expected size and is a JPEG; metadata matches the official split file
    for (config, split), rows in tables.items():
        for i, row in enumerate(rows):
            payload = images[(config, split)][i]["image"]["bytes"]
            picture = Image.open(io.BytesIO(payload))
            picture.load()
            want = (row["size"],) * 2 if config == "fields" else (CANDIDATE_OUT,) * 2
            check(
                payload[:3] == b"\xff\xd8\xff" and picture.size == want and picture.mode == "RGB",
                f"{config}/{split}[{i}] image {picture.size} {picture.mode}",
            )
            tumor, species = TUMORS[official.loc[row["slide"], "Tumor"]]
            check(row["tumor_type"] == tumor, f"{config}/{split}[{i}] tumor_type {row['tumor_type']}")
            check(row["species"] == species, f"{config}/{split}[{i}] species")

    # fields: coordinates recomputed from the source annotations
    for split in SPLITS:
        for row in tables[("fields", split)]:
            x0, y0, size = row["x0"], row["y0"], row["size"]
            width, height = sizes[row["slide"]]
            check(size == FIELD_SIZE, f"{row['field_id']}: size {size}")
            check(0 <= x0 <= width - size and 0 <= y0 <= height - size, f"{row['field_id']}: field leaves the slide")
            inner = {1: [], 2: []}
            for a in annotations[row["slide"]].values():
                dx, dy = a["x"] - x0, a["y"] - y0
                in_inner = EDGE <= dx < size - EDGE and EDGE <= dy < size - EDGE
                in_outer = -EDGE <= dx < size + EDGE and -EDGE <= dy < size + EDGE
                if in_inner:
                    inner[a["cls"]].append((round(dx, 1), round(dy, 1)))
                    if row["kind"] == "dense":
                        allowed = {(1, 1)} if a["cls"] == 1 else {(2, 2), (2, 2, 2)}
                        check(tuple(a["votes"]) in allowed, f"{row['field_id']}: disputed cell in a dense field")
                check(
                    not (a["cls"] == 1 and in_outer and not in_inner),
                    f"{row['field_id']}: mitotic figure in border band",
                )
            for cls, key in ((1, "mitoses"), (2, "imposters")):
                stored = sorted((round(x, 1), round(y, 1)) for x, y in row[key])
                check(
                    stored == sorted(inner[cls]),
                    f"{row['field_id']}: {key} differ from source ({len(stored)} vs {len(inner[cls])})",
                )
        by_slide = defaultdict(list)
        for row in tables[("fields", split)]:
            by_slide[row["slide"]].append(row)
        for members in by_slide.values():
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    overlap = abs(left["x0"] - right["x0"]) < FIELD_SIZE and abs(left["y0"] - right["y0"]) < FIELD_SIZE
                    check(not overlap, f"overlapping fields: {left['field_id']} and {right['field_id']}")

    # candidates: label/votes/coordinates recomputed from the source annotations; crop stays inside the slide
    for split in SPLITS:
        for row in tables[("candidates", split)]:
            slide, annotation_id = row["candidate_id"][1:].split("_a")
            source = annotations[int(slide)].get(int(annotation_id))
            check(source is not None and int(slide) == row["slide"], f"{row['candidate_id']}: not in source")
            if source:
                check(CLASS_NAMES[source["cls"]] == row["label"], f"{row['candidate_id']}: label")
                check(",".join(map(str, source["votes"])) == row["votes"], f"{row['candidate_id']}: votes")
                check(
                    (round(source["x"]), round(source["y"])) == (row["x"], row["y"]),
                    f"{row['candidate_id']}: coordinates",
                )
            width, height = sizes[row["slide"]]
            half = CANDIDATE_NATIVE // 2
            check(
                half <= row["x"] <= width - half and half <= row["y"] <= height - half,
                f"{row['candidate_id']}: crop leaves the slide",
            )

    # Dense fields meet the hard-task criterion; mixed fields retain their source-sampling contract.
    for split in SPLITS:
        for row in tables[("fields", split)]:
            check(row["kind"] in ("dense", "anchored", "mixed", "random"), f"{row['field_id']}: kind {row['kind']!r}")
            if row["kind"] == "dense":
                check(len(row["mitoses"]) >= 3, f"{row['field_id']}: dense field has fewer than three figures")
            if row["kind"] == "mixed":
                m, i = len(row["mitoses"]), len(row["imposters"])
                check(
                    m >= 1 and i >= 1 and 2 <= m + i <= MAX_CELLS, f"{row['field_id']}: mixed field with {m}+{i} cells"
                )

    if args.spot_check:
        print(f"\npixel spot-check ({args.spot_check} slides re-downloaded and re-cropped):")
        spot_check(work, tables, images, pick_spot_check_slides(slides, args.spot_check), check)

    seen: set[str] = set()
    benchmark: dict[str, set[str]] = {}
    for config in ("tasks", "warmup"):
        if (root / config).exists():
            check_tasks(root, report, raw, tables, images, scoring, check, config, seen, benchmark)
        else:
            check(False, f"the dataset has no {config} config")
    test_fields = tables[("fields", "test")]
    print(
        "\ntest fields: mitoses per field",
        dict(sorted(Counter(min(len(r["mitoses"]), 10) for r in test_fields).items())),
        "(10 = 10+)",
    )

    if problems:
        print(f"\n{len(problems)} PROBLEMS, first 50:")
        for message in problems[:50]:
            print("  -", message)
        return 1
    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
