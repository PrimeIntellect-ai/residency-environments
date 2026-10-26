# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=4.0",
#     "huggingface_hub>=1.0",
#     "imagecodecs",
#     "numpy>=2.1",
#     "pandas>=2.2",
#     "pillow>=11.0",
#     "pyarrow>=17.0",
#     "tifffile",
# ]
# ///
"""Build the ``midog-mitosis`` dataset from MIDOG++ (CC0) as a four-config HuggingFace dataset.

Source: Aubreville et al., "A comprehensive multi-domain dataset for mitotic figure detection",
Scientific Data 10, 484 (2023), https://doi.org/10.1038/s41597-023-02327-4. 503 region-of-interest slides
(2 mm^2 each, 40x, 0.23-0.25 um/px) from 7 tumor types of humans and dogs, with 11,937 consensus mitotic
figures and 14,349 imposters. Everything is fetched from Figshare (CC0, no login).

Each 160 MB slide is downloaded, cut into fields and candidate crops, and deleted, so the ~75 GB source
never sits on disk. Finished slides are staged under ``--work-dir``; an interrupted build resumes.

    uv run generators/midog-mitosis/build_dataset.py --work-dir ~/midog-build

Output (``<work-dir>/dataset``) is a complete HF dataset repository: parquet shards for the ``tasks`` (the benchmark:
image, question, expected answer, rubric), ``fields`` and ``candidates`` (the source tables the tasks are cut from)
configs, a dataset card, and ``build_report.json``. A ``--slides`` smoke run writes ``dataset-partial`` instead and is
flagged ``partial`` in the report. ``--tasks-only`` re-plans the tasks from an existing build without touching slides.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import pyarrow.parquet as pq
import tifffile
from PIL import Image

FIGSHARE_COLLECTION = 6615571
ANNOTATIONS_FILE = 41265615  # MIDOG++.json
SPLIT_FILE = 41332611  # datasets_xvalidation.csv
FIGSHARE_API = "https://api.figshare.com/v2"
DOWNLOAD = "https://ndownloader.figshare.com/files/{id}"
USER_AGENT = "midog-mitosis-build/1.0 (open-source dataset build; github.com/dylantirandaz)"
"""Figshare's load balancer rejects urllib's default User-Agent with HTTP 403."""

FIELD_SIZE = 768
"""Field edge in native pixels (~192 um at 0.25 um/px)."""
FIELD_EDGE = 40
"""Objects whose center is closer than this to a field border are not shown or scored."""
FIELDS_ANCHORED = 2
FIELDS_MIXED = 3
FIELDS_RANDOM = 2
MIXED_TRIES = 600
RANDOM_TRIES = 200
MIXED_MAX_CANDIDATES = 12
"""A mixed field holds at least one mitotic figure and one imposter and at most this many annotated cells."""
STAGE_VERSION = 3
"""Bump when sampling or cropping logic changes in a way the constants in ``stage_settings`` do not capture."""
CANDIDATE_NATIVE = 128
CANDIDATE_OUT = 512
CANDIDATES_PER_CLASS = 8
JPEG_QUALITY = 92
SPLIT_SALT = "midog-mitosis-hard-v1"
MIN_TARGETS = 3
TASK_LIMITS = {"test": 512, "validation": 64}
DENSE_SEED = 97000
DENSE_OFFSETS = 128
DENSE_GRID = 192
DENSE_TRIALS = 12
CORE_REPO = "tirandazdylan/midog-mitosis"
CORE_REVISION = "16a511cd86559c6c1d28c1a256f26a455df78a2a"
CORE_SHA256 = {
    "fields": "53eae432700c5d7e289671f7c9c7600b4764efada21083f2957cd05cc24e8768",
    "tasks": "333ca8cc1f819c187bb96a62bc4f2af3458a497f4c8ee81970d3045ee6b3524b",
}

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
CLASSES = {1: "mitotic figure", 2: "imposter"}


# --------------------------------------------------------------------------------------
# Source metadata
# --------------------------------------------------------------------------------------
def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def http_get(url: str, retries: int = 8) -> bytes:
    """GET with backoff; a 403/429 from the Figshare API means the rate limit was hit, so wait it out."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request(url), timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (403, 429, 500, 502, 503) or attempt == retries - 1:
                raise
            time.sleep(60 * (attempt + 1) if error.code in (403, 429) else 2**attempt)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def figshare_files(work: Path) -> dict[str, dict]:
    """``{file_name: {id, md5, size}}`` for every slide TIFF in the MIDOG++ collection.

    One API call per slide, paced at 1 request/s (Figshare asks API users to stay under that) and cached
    incrementally, so an interrupted crawl resumes instead of starting over.
    """
    cache = work / "manifest.json"
    if cache.exists():
        return json.loads(cache.read_text())
    # Paged listings of this collection are not a stable snapshot (articles repeat and others are skipped
    # across pages), so take everything in one request and de-duplicate.
    listing = json.loads(http_get(f"{FIGSHARE_API}/collections/{FIGSHARE_COLLECTION}/articles?page=1&page_size=1000"))
    ids = list(dict.fromkeys(article["id"] for article in listing))
    time.sleep(1.0)
    partial = work / "manifest.partial.json"
    seen: dict[str, list[dict]] = json.loads(partial.read_text()) if partial.exists() else {}
    for n, article_id in enumerate(ids, 1):
        if str(article_id) in seen:
            continue
        seen[str(article_id)] = json.loads(http_get(f"{FIGSHARE_API}/articles/{article_id}"))["files"]
        time.sleep(1.0)
        if n % 25 == 0:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(seen))
            print(f"manifest {n}/{len(ids)}", flush=True)
    manifest = {
        f["name"]: {"id": f["id"], "md5": f["computed_md5"], "size": f["size"]}
        for files in seen.values()
        for f in files
        if f["name"].endswith(".tiff")
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    partial.unlink(missing_ok=True)
    return manifest


DOWNLOAD_ATTEMPTS = 8


def download(url: str, dest: Path, md5: str) -> None:
    """Fetch ``url`` to ``dest`` and verify its MD5.

    Rides out network and DNS outages (waits grow to 2 minutes) and rate limits (403/429 wait a minute or more),
    fails fast on 404, and never leaves a partial or corrupt file behind.
    """
    try:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            digest = hashlib.md5()
            try:
                with urllib.request.urlopen(request(url), timeout=120) as response, dest.open("wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
            except urllib.error.HTTPError as error:
                if error.code == 404 or attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                time.sleep(60 * (attempt + 1) if error.code in (403, 429) else min(5 * 2**attempt, 120))
                continue
            except OSError:
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                time.sleep(min(5 * 2**attempt, 120))
                continue
            if digest.hexdigest() == md5:
                return
        raise RuntimeError(f"md5 mismatch for {dest.name} after {DOWNLOAD_ATTEMPTS} attempts")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def load_slides(work: Path) -> tuple[pd.DataFrame, dict[int, list[dict]], dict]:
    """The 503 annotated slides with tumor metadata, split, and their annotations."""
    for name, file_id in (("MIDOG++.json", ANNOTATIONS_FILE), ("datasets_xvalidation.csv", SPLIT_FILE)):
        path = work / "source" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(http_get(DOWNLOAD.format(id=file_id)))
    raw = json.loads((work / "source" / "MIDOG++.json").read_text())
    official = pd.read_csv(work / "source" / "datasets_xvalidation.csv", sep=";")
    images = pd.DataFrame(raw["images"]).rename(columns={"id": "slide"})
    slides = images.merge(official, left_on="slide", right_on="Slide", how="inner")
    assert len(slides) == 503, len(slides)
    assert (slides.tumor_type.map(TUMORS) == slides.Tumor.map(TUMORS)).all(), "tumor labels disagree"
    slides["tumor"] = slides.Tumor.map(lambda t: TUMORS[t][0])
    slides["species"] = slides.Tumor.map(lambda t: TUMORS[t][1])
    slides["scanner"] = slides.Scanner.str.replace("Hamammatsu", "Hamamatsu")
    slides["origin"] = slides.Origin
    annotations: dict[int, list[dict]] = {int(s): [] for s in slides.slide}
    for a in raw["annotations"]:
        if a["image_id"] in annotations:
            x0, y0, x1, y1 = a["bbox"]
            annotations[a["image_id"]].append(
                {"id": a["id"], "cls": a["category_id"], "x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "votes": a["labels"]}
            )
    counts = Counter(c["cls"] for anns in annotations.values() for c in anns)
    assert counts == {1: 11937, 2: 14349}, counts
    return slides.sort_values("slide").reset_index(drop=True), annotations, {"official_class_counts": dict(counts)}


def load_core(work: Path, local: str | None) -> tuple[list[dict], list[dict]]:
    """The 36 retained tasks and their source fields, verified against the immutable input shards."""
    from huggingface_hub import hf_hub_download

    root = work / "retained"
    root.mkdir(parents=True, exist_ok=True)
    for config, expected in CORE_SHA256.items():
        dest = root / f"{config}.parquet"
        if not dest.exists():
            filename = f"{config}/test-00000-of-00001.parquet"
            source = (
                Path(local).expanduser() / filename
                if local
                else Path(hf_hub_download(CORE_REPO, filename, repo_type="dataset", revision=CORE_REVISION))
            )
            shutil.copyfile(source, dest)
        with dest.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{dest}: checksum {actual} differs from retained input {expected}")
    tasks = pq.read_table(root / "tasks.parquet", filters=[("n_targets", ">=", MIN_TARGETS)]).to_pylist()
    assert len(tasks) == 36
    parts = {part for task in tasks for part in task["parts"]}
    fields = pq.read_table(root / "fields.parquet", filters=[("field_id", "in", sorted(parts))]).to_pylist()
    assert {field["field_id"] for field in fields} == parts
    return tasks, fields


def dense_fields(slide: dict, anns: list[dict], locked: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Pack clean, non-overlapping 3+ figure fields using annotations only, never model outcomes."""
    cells = np.array(
        [
            (a["x"], a["y"], a["cls"], tuple(a["votes"]) in (CLEAN_MITOTIC if a["cls"] == 1 else CLEAN_IMPOSTER))
            for a in anns
        ],
        dtype=float,
    ).reshape(-1, 4)
    gold = cells[(cells[:, 2] == 1) & (cells[:, 3] == 1), :2]
    if len(gold) < MIN_TARGETS:
        return []
    rng = np.random.default_rng(DENSE_SEED + int(slide["slide"]))
    offsets = rng.uniform(FIELD_EDGE, FIELD_SIZE - FIELD_EDGE, size=(len(gold), DENSE_OFFSETS, 2))
    corners = (
        np.clip(gold[:, None, :] - offsets, [0, 0], [slide["width"] - FIELD_SIZE, slide["height"] - FIELD_SIZE])
        .astype(int)
        .reshape(-1, 2)
    )
    gx, gy = np.meshgrid(
        np.arange(0, slide["width"] - FIELD_SIZE + 1, DENSE_GRID),
        np.arange(0, slide["height"] - FIELD_SIZE + 1, DENSE_GRID),
    )
    corners = np.unique(np.vstack([corners, np.column_stack([gx.ravel(), gy.ravel()])]), axis=0)
    mitotic = cells[:, 2] == 1
    valid = []
    for start in range(0, len(corners), 512):
        batch = corners[start : start + 512]
        delta = cells[None, :, :2] - batch[:, None, :]
        inner = np.all((delta >= FIELD_EDGE) & (delta < FIELD_SIZE - FIELD_EDGE), axis=2)
        outer = np.all((delta >= -FIELD_EDGE) & (delta < FIELD_SIZE + FIELD_EDGE), axis=2)
        counts = (inner & mitotic).sum(axis=1)
        accepted = (
            (counts >= MIN_TARGETS)
            & ~(inner & (cells[:, 3] == 0)).any(axis=1)
            & ~((outer & ~inner) & mitotic).any(axis=1)
        )
        valid.extend((int(x), int(y), int(n)) for (x, y), n in zip(batch[accepted], counts[accepted]))
    candidates = np.array(valid, dtype=int).reshape(-1, 3)
    best = []
    for trial in range(DENSE_TRIALS):
        selected = []
        rng = np.random.default_rng(int(slide["slide"]) * 100 + trial)
        order = np.lexsort((rng.random(len(candidates)), candidates[:, 2]))
        for index in order:
            x, y, _ = candidates[index]
            if any(abs(int(x) - a) < FIELD_SIZE and abs(int(y) - b) < FIELD_SIZE for a, b, _ in locked + selected):
                continue
            selected.append((int(x), int(y), "dense"))
        if len(selected) > len(best):
            best = selected
    return best


def assign_splits(slides: pd.DataFrame, counts: dict[int, int]) -> dict[str, list[int]]:
    """Retain official test slides; expand test and reserve validation in seeded, tumor-balanced slide order."""
    assigned = {"test": set(slides.loc[slides.Dataset == "test", "slide"]), "validation": set()}
    remaining = set(slides.loc[slides.Dataset == "train", "slide"])

    def ordered(pool: set[int], split: str) -> list[int]:
        groups = []
        for _, group in slides[slides.slide.isin(pool)].groupby("tumor", sort=True):
            members = [int(s) for s in group.slide if counts[int(s)]]
            members.sort(key=lambda s: hashlib.sha256(f"{SPLIT_SALT}:{split}:{s}".encode()).digest(), reverse=True)
            groups.append(members)
        result = []
        while any(groups):
            result.extend(group.pop() for group in groups if group)
        return result

    for split, target in TASK_LIMITS.items():
        available = sum(counts[int(s)] for s in assigned[split])
        for slide in ordered(remaining, split):
            if available >= target:
                break
            assigned[split].add(slide)
            remaining.remove(slide)
            available += counts[slide]
        if available < target:
            raise ValueError(f"Only {available} planned {split} tasks, need {target}")
    assigned["train"] = remaining
    for split, members in assigned.items():
        slides.loc[slides.slide.isin(members), "split"] = split
    return {split: sorted(map(int, members)) for split, members in assigned.items()}


# --------------------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------------------
def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def sample_fields(
    pixels: np.ndarray, anns: list[dict], rng: np.random.Generator, planned: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    """Top-left corners and kinds of the fields to cut from one slide; fields never overlap.

    - ``anchored``: contains a mitotic figure at a uniformly random position inside the field (used for counting and
      localization, which need mitotic figures to be present).
    - ``mixed``: uniformly random position, accepted only if it holds at least one mitotic figure and one imposter
      and at most ``MIXED_MAX_CANDIDATES`` cells.
    - ``random``: uniformly random tissue.

    ``mixed`` and ``random`` positions are drawn independently of the class of any cell. A field is rejected when a
    mitotic figure sits within ``FIELD_EDGE`` px of its border, because the figure would be partially visible and its
    gold label ambiguous.
    """
    height, width = pixels.shape[:2]
    mitoses = [(a["x"], a["y"]) for a in anns if a["cls"] == 1]
    imposters = [(a["x"], a["y"]) for a in anns if a["cls"] == 2]
    fields = list(planned)

    def visible(points: list[tuple[float, float]], x0: int, y0: int) -> int:
        return sum(
            FIELD_EDGE <= x - x0 < FIELD_SIZE - FIELD_EDGE and FIELD_EDGE <= y - y0 < FIELD_SIZE - FIELD_EDGE
            for x, y in points
        )

    def acceptable(x0: int, y0: int) -> bool:
        if any(abs(x0 - cx) < FIELD_SIZE and abs(y0 - cy) < FIELD_SIZE for cx, cy, _ in fields):
            return False
        for mx, my in mitoses:
            in_outer = (
                x0 - FIELD_EDGE <= mx < x0 + FIELD_SIZE + FIELD_EDGE
                and y0 - FIELD_EDGE <= my < y0 + FIELD_SIZE + FIELD_EDGE
            )
            in_inner = (
                x0 + FIELD_EDGE <= mx < x0 + FIELD_SIZE - FIELD_EDGE
                and y0 + FIELD_EDGE <= my < y0 + FIELD_SIZE - FIELD_EDGE
            )
            if in_outer and not in_inner:
                return False
        return True

    def uniform_corner() -> tuple[int, int]:
        return int(rng.integers(0, width - FIELD_SIZE)), int(rng.integers(0, height - FIELD_SIZE))

    added = 0
    for i in rng.permutation(len(mitoses)):
        if added >= FIELDS_ANCHORED:
            break
        for _ in range(8):
            offset = rng.uniform(FIELD_EDGE, FIELD_SIZE - FIELD_EDGE, size=2)
            x0 = int(np.clip(mitoses[i][0] - offset[0], 0, width - FIELD_SIZE))
            y0 = int(np.clip(mitoses[i][1] - offset[1], 0, height - FIELD_SIZE))
            if acceptable(x0, y0) and visible(mitoses, x0, y0) >= 1:
                fields.append((x0, y0, "anchored"))
                added += 1
                break
    added = 0
    for _ in range(MIXED_TRIES):
        if added >= FIELDS_MIXED:
            break
        x0, y0 = uniform_corner()
        m, i = visible(mitoses, x0, y0), visible(imposters, x0, y0)
        if m >= 1 and i >= 1 and m + i <= MIXED_MAX_CANDIDATES and acceptable(x0, y0):
            fields.append((x0, y0, "mixed"))
            added += 1
    added = 0
    for _ in range(RANDOM_TRIES):
        if added >= FIELDS_RANDOM:
            break
        x0, y0 = uniform_corner()
        patch = pixels[y0 : y0 + FIELD_SIZE : 8, x0 : x0 + FIELD_SIZE : 8]
        if (patch.min(axis=-1) > 215).mean() > 0.35:  # mostly glass
            continue
        if acceptable(x0, y0):
            fields.append((x0, y0, "random"))
            added += 1
    return fields


def process_slide(job: tuple[dict, list[dict], str, str, dict, list, list]) -> None:
    slide, anns, work_dir, stage_dir, source, planned, retained = job
    work, stage = Path(work_dir), Path(stage_dir)
    number = int(slide["slide"])
    tiff = work / "raw" / slide["file_name"]
    tiff.parent.mkdir(parents=True, exist_ok=True)
    download(DOWNLOAD.format(id=source["id"]), tiff, source["md5"])
    try:
        pixels = tifffile.imread(tiff)[..., :3]
    finally:
        tiff.unlink()
    height, width = pixels.shape[:2]
    assert (width, height) == (slide["width"], slide["height"]), (number, pixels.shape)
    rng = np.random.default_rng(int(hashlib.sha256(f"{SPLIT_SALT}:{number}".encode()).hexdigest()[:8], 16))
    common = {
        "slide": number,
        "tumor_type": slide["tumor"],
        "species": slide["species"],
        "scanner": slide["scanner"],
        "origin": slide["origin"],
        "split": slide["split"],
    }
    for sub in ("fields", "candidates"):
        (stage / sub).mkdir(parents=True, exist_ok=True)

    field_ids = []
    retained_by_corner = {(field["x0"], field["y0"]): field for field in retained}
    for x0, y0, kind in sample_fields(pixels, anns, rng, planned):
        original = retained_by_corner.get((x0, y0))
        digest = hashlib.sha256(f"{x0}:{y0}".encode()).hexdigest()[:12]
        field_id = original["field_id"] if original else f"s{number:04d}_f{digest}"
        inside = [
            a
            for a in anns
            if FIELD_EDGE <= a["x"] - x0 < FIELD_SIZE - FIELD_EDGE
            and FIELD_EDGE <= a["y"] - y0 < FIELD_SIZE - FIELD_EDGE
        ]
        meta = {
            **common,
            "field_id": field_id,
            "x0": x0,
            "y0": y0,
            "size": FIELD_SIZE,
            "kind": kind,
            "mitoses": [[round(a["x"] - x0, 1), round(a["y"] - y0, 1)] for a in inside if a["cls"] == 1],
            "imposters": [[round(a["x"] - x0, 1), round(a["y"] - y0, 1)] for a in inside if a["cls"] == 2],
        }
        payload = (
            original["image"]["bytes"]
            if original
            else jpeg(Image.fromarray(pixels[y0 : y0 + FIELD_SIZE, x0 : x0 + FIELD_SIZE]))
        )
        (stage / "fields" / f"{field_id}.jpg").write_bytes(payload)
        (stage / "fields" / f"{field_id}.json").write_text(json.dumps(meta))
        field_ids.append(field_id)

    half = CANDIDATE_NATIVE // 2
    candidate_ids = []
    usable = {
        cls: [a for a in anns if a["cls"] == cls and half <= a["x"] < width - half and half <= a["y"] < height - half]
        for cls in (1, 2)
    }
    # equal numbers of both classes from every slide, so the class of a candidate cannot be guessed from its slide
    per_class = min(CANDIDATES_PER_CLASS, len(usable[1]), len(usable[2]))
    for cls in (1, 2):
        for pick in rng.permutation(len(usable[cls]))[:per_class]:
            a = usable[cls][pick]
            cx, cy = int(round(a["x"])), int(round(a["y"]))
            crop = Image.fromarray(pixels[cy - half : cy + half, cx - half : cx + half])
            crop = crop.resize((CANDIDATE_OUT, CANDIDATE_OUT), Image.BICUBIC)
            candidate_id = f"s{number:04d}_a{a['id']}"
            meta = {
                **common,
                "candidate_id": candidate_id,
                "label": CLASSES[cls],
                "x": cx,
                "y": cy,
                "votes": ",".join(map(str, a["votes"])),
            }
            (stage / "candidates" / f"{candidate_id}.jpg").write_bytes(jpeg(crop))
            (stage / "candidates" / f"{candidate_id}.json").write_text(json.dumps(meta))
            candidate_ids.append(candidate_id)
    (stage / "slides").mkdir(parents=True, exist_ok=True)
    marker = stage / "slides" / f"{number:04d}.json"
    marker.with_suffix(".tmp").write_text(json.dumps({"fields": field_ids, "candidates": candidate_ids}))
    marker.with_suffix(".tmp").replace(marker)


# --------------------------------------------------------------------------------------
# Orchestration and assembly
# --------------------------------------------------------------------------------------
def stage_settings() -> dict:
    """Constants that shape staged slides. A stage built with other values must not be mixed with this code."""
    return {
        "stage_version": STAGE_VERSION,
        "field_size": FIELD_SIZE,
        "field_edge": FIELD_EDGE,
        "fields_anchored": FIELDS_ANCHORED,
        "fields_mixed": FIELDS_MIXED,
        "fields_random": FIELDS_RANDOM,
        "mixed_tries": MIXED_TRIES,
        "random_tries": RANDOM_TRIES,
        "mixed_max_candidates": MIXED_MAX_CANDIDATES,
        "candidate_native": CANDIDATE_NATIVE,
        "candidate_out": CANDIDATE_OUT,
        "candidates_per_class": CANDIDATES_PER_CLASS,
        "jpeg_quality": JPEG_QUALITY,
        "split_salt": SPLIT_SALT,
        "min_targets": MIN_TARGETS,
        "task_limits": TASK_LIMITS,
        "dense_seed": DENSE_SEED,
        "dense_offsets": DENSE_OFFSETS,
        "dense_grid": DENSE_GRID,
        "dense_trials": DENSE_TRIALS,
        "retained_dataset": CORE_REPO,
        "retained_revision": CORE_REVISION,
    }


def run_build(args: argparse.Namespace) -> None:
    work = Path(args.work_dir).expanduser()
    stage = work / "stage"
    manifest = figshare_files(work)
    slides, annotations, provenance = load_slides(work)
    retained_tasks, retained_fields = load_core(work, args.retained_dataset)
    core_by_slide = defaultdict(list)
    for field in retained_fields:
        core_by_slide[field["slide"]].append(field)
    plans = {}
    counts = Counter(task["slide"] for task in retained_tasks)
    for slide in slides.to_dict("records"):
        number = int(slide["slide"])
        locked = [(f["x0"], f["y0"], f["kind"]) for f in core_by_slide[number]]
        dense = dense_fields(slide, annotations[number], locked)
        plans[number] = locked + dense
        counts[number] += len(dense)
    assignments = assign_splits(slides, counts)
    provenance["split_assignments"] = assignments
    provenance["retained_tasks"] = [
        {key: task[key] for key in ("task_id", "parts", "slide")}
        | {"image_sha256": hashlib.sha256(task["image"]["bytes"]).hexdigest()}
        for task in retained_tasks
    ]
    print(
        f"planned tasks={sum(counts.values())}; slides by split={ {k: len(v) for k, v in assignments.items()} }",
        flush=True,
    )
    missing = [n for n in slides.file_name if n not in manifest]
    assert not missing, f"slides missing from Figshare manifest: {missing[:5]}"
    stage.mkdir(parents=True, exist_ok=True)
    marker = stage / "settings.json"
    if marker.exists() and json.loads(marker.read_text()) != stage_settings():
        raise SystemExit(f"{stage} holds slides built with different settings; delete it or use another --work-dir")
    marker.write_text(json.dumps(stage_settings(), indent=1, sort_keys=True))
    partial = bool(args.slides)
    if partial:
        slides = slides.groupby(["tumor", "split"], group_keys=False).head(args.slides).sort_values("slide")
    pending = [s for _, s in slides.iterrows() if not (stage / "slides" / f"{int(s.slide):04d}.json").exists()]
    print(f"slides={len(slides)} pending={len(pending)}", flush=True)
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(
                process_slide,
                (
                    s.to_dict(),
                    annotations[int(s.slide)],
                    str(work),
                    str(stage),
                    manifest[s.file_name],
                    plans[int(s.slide)],
                    core_by_slide[int(s.slide)],
                ),
            )
            for s in pending
        ]
        try:
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done % 10 == 0 or done == len(futures):
                    rate = done / (time.time() - started)
                    print(f"slide {done}/{len(futures)} eta={(len(futures) - done) / rate / 60:.1f}min", flush=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    provenance |= {
        "slide_md5": {name: manifest[name]["md5"] for name in slides.file_name},
        "source_sha256": {
            name: hashlib.sha256((work / "source" / name).read_bytes()).hexdigest()
            for name in ("MIDOG++.json", "datasets_xvalidation.csv")
        },
    }
    assemble(slides, stage, work / ("dataset-partial" if partial else "dataset"), provenance, partial)


def assemble(slides: pd.DataFrame, stage: Path, out: Path, provenance: dict, partial: bool) -> None:
    import datasets
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HFImage

    common = {
        "slide": Value("int32"),
        "tumor_type": Value("string"),
        "species": Value("string"),
        "scanner": Value("string"),
        "origin": Value("string"),
    }
    configs = {
        "fields": Features(
            {
                "field_id": Value("string"),
                **common,
                "x0": Value("int32"),
                "y0": Value("int32"),
                "size": Value("int32"),
                "kind": Value("string"),
                "mitoses": Sequence(Sequence(Value("float32"), length=2)),
                "imposters": Sequence(Sequence(Value("float32"), length=2)),
                "image": HFImage(),
            }
        ),
        "candidates": Features(
            {
                "candidate_id": Value("string"),
                **common,
                "label": Value("string"),
                "x": Value("int32"),
                "y": Value("int32"),
                "votes": Value("string"),
                "image": HFImage(),
            }
        ),
    }
    # from_generator caches by a fingerprint of the generator, not of the bytes it reads: keep the cache private and
    # empty so a rebuild can never ship images from an earlier one.
    cache = out.parent / "hf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    if out.exists():
        shutil.rmtree(out)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "tifffile": tifffile.__version__,
        "datasets": datasets.__version__,
    }
    report: dict = {"partial": partial, "configs": {}, "provenance": provenance | {"versions": versions}}
    report["stats"] = {
        "slides": len(slides),
        "test_slides": int((slides.split == "test").sum()),
        "tumor_types": int(slides.tumor.nunique()),
        "scanners": int(slides.scanner.nunique()),
        "laboratories": int(slides.origin.nunique()),
    }
    by_split = {split: sorted(int(s) for s in group.slide) for split, group in slides.groupby("split")}
    for config, features in configs.items():
        (out / config).mkdir(parents=True)
        report["configs"][config] = {}
        for split, split_slides in by_split.items():
            keys = []
            for slide in split_slides:
                index = json.loads((stage / "slides" / f"{slide:04d}.json").read_text())
                keys += index[config]
            random.Random(f"{SPLIT_SALT}:rows:{config}:{split}").shuffle(keys)
            rows = {k: json.loads((stage / config / f"{k}.json").read_text()) for k in keys}

            def generate(keys=keys, rows=rows, config=config):
                for k in keys:
                    meta = rows[k]
                    yield {
                        **{name: meta[name] for name in configs[config] if name != "image"},
                        "image": {"bytes": (stage / config / f"{k}.jpg").read_bytes(), "path": None},
                    }

            dataset = Dataset.from_generator(generate, features=features, cache_dir=str(cache))
            nbytes = sum((stage / config / f"{k}.jpg").stat().st_size for k in keys)
            shards = max(1, -(-nbytes // (300 * 1024**2)))
            for i in range(shards):
                dataset.shard(num_shards=shards, index=i, contiguous=True).to_parquet(
                    out / config / f"{split}-{i:05d}-of-{shards:05d}.parquet"
                )
            frame = pd.DataFrame(rows.values())
            summary = {"rows": len(keys), "slides": len(split_slides), "shards": shards}
            if config == "fields":
                summary |= {
                    "mitoses": int(frame.mitoses.map(len).sum()),
                    "imposters": int(frame.imposters.map(len).sum()),
                    "fields_without_mitoses": int((frame.mitoses.map(len) == 0).sum()),
                    "by_kind": dict(Counter(frame.kind)),
                    "by_tumor": dict(Counter(frame.tumor_type)),
                }
            else:
                summary |= {"by_label": dict(Counter(frame.label)), "by_tumor": dict(Counter(frame.tumor_type))}
            report["configs"][config][split] = summary
    shutil.rmtree(cache, ignore_errors=True)
    report["settings"] = stage_settings()
    build_tasks(out, report)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out / "README.md").write_text(dataset_card(report))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2, sort_keys=True))


# --------------------------------------------------------------------------------------
# Tasks: one row per benchmark task, holding the image shown, the question, the expected answer and the rubric
# --------------------------------------------------------------------------------------
SPLITS = ("train", "validation", "test")
CONFIGS = ("tasks", "warmup")
"""``tasks`` holds at least three mitotic figures per image. ``warmup`` holds exactly one figure per field."""
TASK_SEED = 0
"""Seeds task selection and ordering; selection never uses a model's performance."""
POINT_SCALE = 1000.0
MATCH_RADIUS_PX = 30
"""7.5 um at 0.25 um/px, the detection criterion of the MIDOG challenges. Only quoted in the rubric text; the reward
is computed by ``scoring.py`` in the environment, and the validator checks the two agree."""
CLEAN_MITOTIC = {(1, 1)}
CLEAN_IMPOSTER = {(2, 2), (2, 2, 2)}
"""Expert votes (1 = mitotic figure, 2 = imposter) of a cell nobody disputed. A task keeps only fields whose annotated
cells are all undisputed, so its expected answer is not a coin flip between experts that a third one settled."""


def load_votes(work: Path) -> dict[tuple[int, float, float], tuple[int, ...]]:
    """Expert votes of every annotation by (slide, center x, center y) in slide pixels, from the cached MIDOG++.json."""
    votes = {}
    for a in json.loads((work / "source" / "MIDOG++.json").read_text())["annotations"]:
        x0, y0, x1, y1 = a["bbox"]
        votes[(a["image_id"], (x0 + x1) / 2, (y0 + y1) / 2)] = tuple(a["labels"])
    return votes


def clean(row: dict, votes: dict) -> bool:
    """Whether every annotated cell of a field is undisputed."""

    def cells(points: list) -> list[tuple[int, ...]]:
        return [votes[(row["slide"], row["x0"] + x, row["y0"] + y)] for x, y in points]

    return all(v in CLEAN_MITOTIC for v in cells(row["mitoses"])) and all(
        v in CLEAN_IMPOSTER for v in cells(row["imposters"])
    )


def plan_tasks(rows: list[dict], votes: dict, split: str, report: dict) -> list[list[int]]:
    """Keep the retained test tasks, then select clean 3+ fields without reusing a source field."""
    where = {row["field_id"]: i for i, row in enumerate(rows)}
    retained = []
    if split == "test":
        for task in report["provenance"]["retained_tasks"]:
            if all(part in where for part in task["parts"]):
                retained.append([where[part] for part in task["parts"]])
    used = {index for task in retained for index in task}
    eligible = [
        [i] for i, row in enumerate(rows) if i not in used and len(row["mitoses"]) >= MIN_TARGETS and clean(row, votes)
    ]
    rng = random.Random(f"{TASK_SEED}:{split}")
    rng.shuffle(eligible)
    if split in TASK_LIMITS and not report["partial"]:
        needed = TASK_LIMITS[split] - len(retained)
        if len(eligible) < needed:
            raise ValueError(f"{split}: {len(eligible)} eligible fields, need {needed}")
        eligible = eligible[:needed]
    warmup = [[i] for i, row in enumerate(rows) if len(row["mitoses"]) == 1 and clean(row, votes)]
    tasks = retained + eligible + warmup
    rng.shuffle(tasks)
    # A field shown inside a benchmark pair is not also a warmup row; filtering after the shuffle keeps every other order.
    paired = {i for task in retained if len(task) > 1 for i in task}
    return [task for task in tasks if len(task) > 1 or task[0] not in paired]


def question(width: int, height: int, parts: int) -> str:
    """The exact text shown with a task's image."""
    if parts == 1:
        shown = f"a {width}×{height} pixel H&E-stained tumor field"
    else:
        shown = f"a {width}×{height} pixel image of two {height}×{height} pixel H&E-stained tumor fields of one slide, side by side,"
    return (
        f"This is {shown} at 40x magnification (0.25 µm per pixel). It contains at least one mitotic figure. "
        "Find every mitotic figure and give the center of each as [x, y], with coordinates normalized to 0-1000 "
        "(x across the image width, y down the image height, [0, 0] at the top-left corner). "
        "The answer counts only if it lists every mitotic figure and nothing else.\n\n"
        'End your reply with "Answer: [[x, y], ...]".'
    )


def rubric(targets: int) -> str:
    """How the reward of this task is computed (the reward itself is ``scoring.py``)."""
    return (
        f"Reward 1 only if the listed points and the {targets} annotated mitotic figure center{'s' if targets > 1 else ''} "
        f"match one-to-one within {MATCH_RADIUS_PX} px (7.5 µm), none missing and none extra; otherwise reward 0. "
        "An answer that cannot be read scores 0."
    )


def make_task(parts: list[dict], images: list[bytes], rank: int) -> dict:
    """One task row from one field, or from two fields of one slide placed side by side (the first on the left)."""
    size = parts[0]["size"]
    width = size * len(parts)
    points = sorted((x + size * k, y) for k, row in enumerate(parts) for x, y in row["mitoses"])
    if len(parts) == 1:
        image = images[0]
    else:
        canvas = Image.new("RGB", (width, size))
        for k, data in enumerate(images):
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"), (size * k, 0))
        image = jpeg(canvas)
    source_id = "+".join(row["field_id"] for row in parts)
    answer = "[" + ", ".join(f"[{x * POINT_SCALE / width:.1f}, {y * POINT_SCALE / size:.1f}]" for x, y in points) + "]"
    return {
        "task_id": f"find:{source_id}",
        "image": {"bytes": image, "path": None},
        "question": question(width, size, len(parts)),
        "answer": answer,
        "rubric": rubric(len(points)),
        "points": [list(point) for point in points],
        "width": width,
        "height": size,
        "n_targets": len(points),
        "n_imposters": sum(len(row["imposters"]) for row in parts),
        "parts": [row["field_id"] for row in parts],
        "rank": rank,
        "source_id": source_id,
        **{k: parts[0][k] for k in ("slide", "tumor_type", "species", "scanner", "origin")},
    }


def field_table(root: Path, split: str):
    """The ``fields`` rows of a split without their images, and the images (as bytes) by row index."""
    from datasets import Dataset
    from datasets import Image as HFImage

    files = sorted(str(f) for f in (root / "fields").glob(f"{split}-*.parquet"))
    dataset = Dataset.from_parquet(files, cache_dir=str(root.parent / "hf-cache"))
    rows = dataset.select_columns([c for c in dataset.column_names if c != "image"]).to_list()
    return rows, dataset.select_columns(["image"]).cast_column("image", HFImage(decode=False))


def task_rows(root: str, split: str, plan: list[tuple[int, list[int]]]):
    """Generator for ``Dataset.from_generator``: module-level and fed only small arguments, so it hashes quickly.

    ``from_generator`` splits a list-valued argument into shards, so every element of the plan carries its own rank.
    """
    path = Path(root)
    rows, images = field_table(path, split)
    retained = {}
    if split == "test":
        retained = {
            task["task_id"]: task
            for task in pq.read_table(
                path.parent / "retained" / "tasks.parquet", filters=[("n_targets", ">=", MIN_TARGETS)]
            ).to_pylist()
        }
    for rank, indices in plan:
        task_id = "find:" + "+".join(rows[i]["field_id"] for i in indices)
        if task_id in retained:
            yield retained[task_id] | {"rank": rank}
        else:
            yield make_task([rows[i] for i in indices], [images[i]["image"]["bytes"] for i in indices], rank)


def in_config(config: str, parts: int, figures: int) -> bool:
    """Benchmark images have 3+ figures; warmup images are single fields with exactly one."""
    return figures >= MIN_TARGETS if config == "tasks" else parts == 1 and figures == 1


def build_tasks(out: Path, report: dict) -> None:
    """Plan and write the ``tasks`` and ``warmup`` configs from the ``fields`` table in ``out``; records the counts in ``report``."""
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HFImage

    features = Features(
        {
            "task_id": Value("string"),
            "image": HFImage(),
            "question": Value("string"),
            "answer": Value("string"),
            "rubric": Value("string"),
            "points": Sequence(Sequence(Value("float32"), length=2)),
            "width": Value("int32"),
            "height": Value("int32"),
            "n_targets": Value("int32"),
            "n_imposters": Value("int32"),
            "parts": Sequence(Value("string")),
            "rank": Value("int32"),
            "source_id": Value("string"),
            "slide": Value("int32"),
            "tumor_type": Value("string"),
            "species": Value("string"),
            "scanner": Value("string"),
            "origin": Value("string"),
        }
    )
    cache = out.parent / "hf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    votes = load_votes(out.parent)
    for config in CONFIGS:
        shutil.rmtree(out / config, ignore_errors=True)
        (out / config).mkdir()
        summary = {}
        for split in SPLITS:
            rows, _ = field_table(out, split)
            kept = [
                t
                for t in plan_tasks(rows, votes, split, report)
                if in_config(config, len(t), sum(len(rows[i]["mitoses"]) for i in t))
            ]
            dataset = Dataset.from_generator(
                task_rows,
                features=features,
                cache_dir=str(cache),
                gen_kwargs={"root": str(out), "split": split, "plan": list(enumerate(kept))},
            )
            shards = max(1, -(-dataset.data.nbytes // (300 * 1024**2)))
            for i in range(shards):
                dataset.shard(num_shards=shards, index=i, contiguous=True).to_parquet(
                    out / config / f"{split}-{i:05d}-of-{shards:05d}.parquet"
                )
            frame = dataset.select_columns(["n_targets", "slide", "parts", "points", "width"]).to_pandas()
            halves = Counter(
                tuple(
                    sorted(
                        ("0", "1", "2+")[min(n, 2)]
                        for n in (sum(x < w / 2 for x, _ in pts), sum(x >= w / 2 for x, _ in pts))
                    )
                )
                for pts, w, parts in zip(frame.points, frame.width, frame.parts)
                if len(parts) == 2
            )
            summary[split] = {
                "rows": len(frame),
                "slides": int(frame.slide.nunique()),
                "shards": shards,
                "by_targets": {str(k): int(v) for k, v in sorted(Counter(frame.n_targets.clip(upper=4)).items())},
                "pairs": int((frame.parts.map(len) == 2).sum()),
                "pair_halves": {"+".join(k): int(v) for k, v in sorted(halves.items())},
            }
        report["configs"][config] = summary
    shutil.rmtree(cache, ignore_errors=True)
    report["task_settings"] = {
        "task_seed": TASK_SEED,
        "minimum_targets": MIN_TARGETS,
        "task_limits": TASK_LIMITS,
        "pairs": "only the retained test pairs; every benchmark field is used once",
        "match_radius_px": MATCH_RADIUS_PX,
        "point_scale": POINT_SCALE,
        "undisputed_mitotic_votes": sorted(map(list, CLEAN_MITOTIC)),
        "undisputed_imposter_votes": sorted(map(list, CLEAN_IMPOSTER)),
    }


CARD = """---
license: cc0-1.0
language:
- en
pretty_name: midog-mitosis
size_categories:
- 1K<n<10K
task_categories:
- visual-question-answering
- object-detection
tags:
- medical
- pathology
- histopathology
- digital-pathology
- mitosis
- mitotic-figure
- veterinary
- verifiers
configs:
- config_name: tasks
  data_files:
  - split: train
    path: tasks/train-*.parquet
  - split: validation
    path: tasks/validation-*.parquet
  - split: test
    path: tasks/test-*.parquet
- config_name: warmup
  data_files:
  - split: train
    path: warmup/train-*.parquet
  - split: validation
    path: warmup/validation-*.parquet
  - split: test
    path: warmup/test-*.parquet
- config_name: fields
  data_files:
  - split: train
    path: fields/train-*.parquet
  - split: validation
    path: fields/validation-*.parquet
  - split: test
    path: fields/test-*.parquet
- config_name: candidates
  data_files:
  - split: train
    path: candidates/train-*.parquet
  - split: validation
    path: candidates/validation-*.parquet
  - split: test
    path: candidates/test-*.parquet
---

# midog-mitosis

H&E tumor fields with consensus mitotic-figure annotations, cut from
[MIDOG++](https://doi.org/10.1038/s41597-023-02327-4) (503 slides, 7 tumor types, humans and dogs). The `tasks` config
is the benchmark: every row is an image holding at least three mitotic figures (one field, or a pair of fields
from one slide), the question asked about it (find every mitotic figure), the expected answer (the coordinates of
each) and the rubric the reward follows. The answer is free response and the reward is binary: the listed points match
the annotated mitotic figures one-to-one, none missing and none extra, or they do not. It is what the `midog-mitosis`
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment serves, as a public benchmark (`test` split) and
an RL environment (`train` / `validation`). The `warmup` config holds the single fields with exactly one figure, in the
same format: they are not part of the benchmark, only training material for models too weak to pass any `tasks` row.
`fields` and `candidates` are the annotated source tables the tasks are cut from.

## Source and license

Everything here derives from **MIDOG++** (Aubreville et al., *Scientific Data* 10, 484, 2023), released by its
authors under **CC0 1.0** on Figshare (collection [10.6084/m9.figshare.c.6615571](https://doi.org/10.6084/m9.figshare.c.6615571)).
This derivative is released under CC0 1.0 as well. Please cite the original work:

> Aubreville, M., Wilm, F., Stathonikos, N., Breininger, K., Donovan, T. A., Jabari, S., Veta, M., Ganz, J., Ammeling, J.,
> van Diest, P. J., Klopfleisch, R., & Bertram, C. A. (2023). A comprehensive multi-domain dataset for mitotic figure
> detection. *Scientific Data*, 10, 484. https://doi.org/10.1038/s41597-023-02327-4

**Changes made:** fields and cell crops were cut from the released slide TIFFs and re-encoded as JPEG (quality {jpeg});
cell crops were upsampled 4x (bicubic); annotations were converted from slide to field coordinates; only slides
with annotations (503 of the 553 in the released JSON) are used. Labels were not altered.

## Configs

**`tasks`** (default, the benchmark) and **`warmup`** have the same columns: one row per task.

| column | meaning |
| --- | --- |
| `task_id` | `find:<source_id>` |
| `image` | the exact image the model sees: a {size} x {size} px field, or two fields of one slide side by side |
| `question` | the exact text sent with the image, answer format included |
| `answer` | the expected final answer: `[[x, y], ...]`, the center of every mitotic figure, normalized to 0-1000 on each axis |
| `rubric` | how the reward of this task is computed |
| `points` | the same centers in image pixels |
| `width`, `height` | image size in pixels (`{size}` x `{size}`, or `{pair_width}` x `{size}` for a pair) |
| `n_targets` | number of mitotic figures in the image (what the model has to find) |
| `n_imposters` | number of annotated non-mitotic look-alikes in the image |
| `parts` | the `field_id`s of the `fields` rows the image was made from (two for a pair, left then right) |
| `rank` | position in the split's fixed pseudo-random order; the environment serves tasks in this order |
| `source_id` | `parts` joined with `+` |
| `slide`, `tumor_type`, `species`, `scanner`, `origin` | slide metadata |

{task_table}

The answers are the MIDOG++ expert annotations, not model output or heuristics. A task keeps only fields in which every
annotated cell is undisputed: the first two experts agreed on every mitotic figure (votes `1,1`) and on every
look-alike (`2,2` or `2,2,2`), so the expected answer is not a coin flip that a third expert settled. A task has at
least three mitotic figures in `tasks` and exactly one in `warmup`, so answering `[]` is never right.
The model is not told how many figures there are, and it
passes only by listing every one and nothing else.

Benchmark images are native-resolution fields or same-slide pairs, each containing at least three figures.
No benchmark field or figure is reused. A field with exactly one figure is a `warmup` row, unless it is one half of a
benchmark pair: no `warmup` row shows a field that a `tasks` row shows.

The 3+ figure criterion is a calibrated difficulty rule, not a preregistered criterion or a performance guarantee.
Dense crop and split selection use expert annotations and fixed seeds, not per-task model outcomes.
Full builds select {test_tasks} test tasks and {validation_tasks} validation tasks in seeded order, after reserving
checksum-pinned seed tasks. Training uses every remaining eligible field on training slides.

The reward is computed by `midog_mitosis/scoring.py` in the environment from the reply and these columns alone.
It requires a one-to-one match to every annotated figure, with none missing or extra, within a 30 px radius.
A fixed single-point answer cannot pass a benchmark task. Partial matches are diagnostic metrics, not partial reward.

**`fields`**: one row per {size} x {size} px field (192 um at 0.25 um/px, 40x).

| column | meaning |
| --- | --- |
| `field_id` | `s<slide>_f<coordinate hash>`, or a checksum-pinned seed field's id |
| `image` | JPEG field |
| `mitoses` | `[[x, y], ...]` field-pixel centers of every consensus mitotic figure in the field |
| `imposters` | same for annotated non-mitotic look-alikes (annotated non-exhaustively by the authors) |
| `slide`, `x0`, `y0`, `size` | source slide id and the field's top-left corner and edge in slide pixels |
| `kind` | how the field was chosen: `dense`, `anchored`, `mixed` or `random` (below) |
| `tumor_type`, `species`, `scanner`, `origin` | slide metadata (lymphoma is the dataset's "lymphosarcoma") |

Fields from one slide never overlap. Fixed seed fields are reserved first, then dense fields are packed; up to
{anchored} anchored, {mixed} mixed and {random} random fields are added in the remaining area:

- `dense`: at least three mitotic figures, every annotated cell undisputed, selected by deterministic non-overlapping packing.
- `anchored`: a mitotic figure sits at a uniformly random position inside the field.
- `mixed`: uniformly random position, kept only if it holds at least one mitotic figure and one imposter and at most
  12 annotated cells.
- `random`: uniformly random tissue.

Objects within {edge} px of a border are not listed, and a field is rejected if a mitotic figure would be cut by its
border, so the lists match what is fully visible.

**`candidates`**: one row per annotated cell, with the same number of mitotic figures and imposters from every slide
(up to {per_class} of each).

| column | meaning |
| --- | --- |
| `candidate_id` | `s<slide>_a<annotation id>` |
| `image` | {native} x {native} px crop centered on the cell, upsampled to {out} x {out} (JPEG) |
| `label` | `mitotic figure` or `imposter` (three-expert consensus) |
| `votes` | the individual expert labels (1 = mitotic figure, 2 = imposter); disagreements were resolved by a third expert |
| `x`, `y`, `slide`, `tumor_type`, `species`, `scanner`, `origin` | as above |

## Usage

```python
from datasets import load_dataset

tasks = load_dataset("__REPO_ID__", "tasks", split="test")
row = tasks[0]  # row["image"], row["question"], row["answer"], row["points"], row["rubric"]
fields = load_dataset("__REPO_ID__", "fields", split="test")
candidates = load_dataset("__REPO_ID__", "candidates", split="test")
```

`build_report.json` in this repository holds the counts, checksums and settings of the build.

## Splits

Splits are by slide (every MIDOG++ slide is a distinct case), including all tasks, warmups, fields and candidate crops.
This is a **custom split**, not the official MIDOG++ split: all 111 official test slides stay in test, and additional
official training slides are held out until there is capacity for {test_tasks} tasks. Remaining slides are assigned to
validation until it can supply {validation_tasks} tasks, then to training. Selection uses seeded, round-robin tumor-type
order and annotation-only crop capacity. The split manifest is recorded in `build_report.json`.

Test covers {test_slides} of {slides} slides ({test_percent}%). **An evaluated model's training slides must be disjoint
from this release's held-out slides.** All configs follow these assignments; using a different training split does not
establish held-out validity. Task order is seeded and field ids do not encode a field's content.

{table}

## Intended use and limitations

Evaluating and training vision-language models on mitotic-figure reading. **Not for clinical use.**

- The answers are public in this dataset. The `midog-mitosis` environment runs every attempt in a fresh sandbox that
  never receives the answer, with outbound traffic blocked except the model route; do the same when you evaluate an
  agent that can execute code, and don't evaluate a model trained on the test slides.
- Mitotic-figure labels are hard: the paper reports that the first two experts disagreed on about 20% of candidates
  (19.98%, its Table 3), resolved by a third. Tasks keep only fields without a disputed annotated cell. Imposters
  were annotated non-exhaustively, and a mitosis the experts missed would count against a model that finds it.
- A pair shows two non-adjacent fields of one slide side by side, as its question states.
- {tumor_types} tumor types, {scanners} scanner models (as named in the official split file), {laboratories}
  laboratories; not a substitute for external validation.

Built by `generators/midog-mitosis/build_dataset.py` (seed `{salt}`). `build_report.json` records the settings, the Figshare
MD5 of every slide, the SHA-256 of the annotation and split files, and the library versions used.
"""


def dataset_card(report: dict) -> str:
    rows = ["| config | split | rows | slides |", "| --- | --- | --- | --- |"]
    for config, splits in report["configs"].items():
        for split in SPLITS:
            if split in splits:
                rows.append(f"| {config} | {split} | {splits[split]['rows']} | {splits[split]['slides']} |")
    tasks = report["configs"]["tasks"]
    task_rows = [
        "| `tasks` (the benchmark) | with 1 mitotic figure | 2 | 3 | 4 or more | of which side-by-side pairs | pair halves (figures left+right, sorted) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        t = tasks[split]
        counts = " | ".join(str(t["by_targets"].get(str(k), 0)) for k in (1, 2, 3, 4))
        halves = ", ".join(f"{k}: {v}" for k, v in t["pair_halves"].items())
        task_rows.append(f"| {split}: {t['rows']} | {counts} | {t['pairs']} | {halves} |")
    settings, stats = report["settings"], report["stats"]
    return CARD.format(
        jpeg=settings["jpeg_quality"],
        size=settings["field_size"],
        pair_width=2 * settings["field_size"],
        anchored=settings["fields_anchored"],
        mixed=settings["fields_mixed"],
        random=settings["fields_random"],
        edge=settings["field_edge"],
        per_class=settings["candidates_per_class"],
        native=settings["candidate_native"],
        out=settings["candidate_out"],
        table="\n".join(rows),
        task_table="\n".join(task_rows),
        salt=settings["split_salt"],
        test_tasks=settings["task_limits"]["test"],
        validation_tasks=settings["task_limits"]["validation"],
        slides=stats["slides"],
        test_slides=stats["test_slides"],
        test_percent=round(100 * stats["test_slides"] / stats["slides"]),
        tumor_types=stats["tumor_types"],
        scanners=stats["scanners"],
        laboratories=stats["laboratories"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default="~/midog-build", help="Staging + output root.")
    parser.add_argument("--slides", type=int, help="Smoke test: only the first N slides of each tumor type and split.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent slide downloads/processors.")
    parser.add_argument("--retained-dataset", help="Local copy of the pinned input dataset; verified by SHA-256.")
    parser.add_argument("--card-only", action="store_true", help="Re-render the dataset card from an existing build.")
    parser.add_argument(
        "--tasks-only", action="store_true", help="Re-plan the tasks (and card) from an existing build."
    )
    args = parser.parse_args()
    out = Path(args.work_dir).expanduser() / ("dataset-partial" if args.slides else "dataset")
    if args.card_only:
        (out / "README.md").write_text(dataset_card(json.loads((out / "build_report.json").read_text())))
        return
    if args.tasks_only:
        report = json.loads((out / "build_report.json").read_text())
        build_tasks(out, report)
        (out / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (out / "README.md").write_text(dataset_card(report))
        return
    run_build(args)


if __name__ == "__main__":
    sys.exit(main())
