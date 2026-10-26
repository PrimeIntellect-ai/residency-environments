# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "torchvision", "datasets>=4.0", "pillow", "numpy", "scipy"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""A small trained specialist on the benchmark: how often does a ResNet-18 detector pass?

    uv run scripts/midog-mitosis/detector.py --dataset ~/midog-build/dataset

An ImageNet-initialised ResNet-18 is fine-tuned on 128 px windows cut from the *train* fields: mitotic figures
(+-6 px jitter) against annotated imposters and plain tissue. At test time it slides over every image of the ``tasks``
config with a fixed stride, keeps the local maxima (30 px non-maximum suppression) above a threshold chosen on the
validation tasks, and is scored with the benchmark's own rule: the listed centers must match the annotated ones
one-to-one within 30 px, none missing and none extra. It never sees a test slide, and it is not told how many figures
there are (the ``oracle`` row is: it keeps the n best peaks).

The numbers are a reference point, not a ceiling: a larger detector trained on the full slides does better. Runs on CPU in
about twenty minutes.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from datasets import Dataset, load_dataset
from datasets import Image as HFImage
from PIL import Image
from scipy.optimize import linear_sum_assignment

SIZE, HALF, RADIUS = 128, 64, 30.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rows(root: str, config: str, split: str):
    path = Path(root).expanduser()
    if path.is_dir():
        return Dataset.from_parquet(sorted(str(file) for file in (path.resolve() / config).glob(f"{split}-*.parquet")))
    return load_dataset(root, config, split=split)


def padded(image: bytes) -> np.ndarray:
    picture = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"), dtype=np.uint8)
    return np.pad(picture, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="reflect")


def window(pad: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """The 128 px window centered on (cx, cy); ``pad`` is shifted by HALF, so the corner is (cx, cy) itself."""
    return pad[int(round(cy)) : int(round(cy)) + SIZE, int(round(cx)) : int(round(cx)) + SIZE]


def training_windows(root: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    fields = load_rows(root, "fields", "train")
    meta = fields.select_columns(["mitoses", "imposters"]).to_list()
    pictures = fields.select_columns(["image"]).cast_column("image", HFImage(decode=False))
    xs, ys = [], []
    for i, m in enumerate(meta):
        pad = padded(pictures[i]["image"]["bytes"])
        cells = [tuple(p) for p in m["mitoses"]] + [tuple(p) for p in m["imposters"]]
        for p in m["mitoses"]:
            for _ in range(2):
                xs.append(window(pad, p[0] + rng.uniform(-6, 6), p[1] + rng.uniform(-6, 6)))
                ys.append(1)
        for p in m["imposters"]:
            xs.append(window(pad, p[0] + rng.uniform(-6, 6), p[1] + rng.uniform(-6, 6)))
            ys.append(0)
        for _ in range(3):
            for _try in range(20):
                cx, cy = rng.uniform(HALF, 768 - HALF), rng.uniform(HALF, 768 - HALF)
                if all(math.hypot(cx - px, cy - py) > 50 for px, py in cells):
                    xs.append(window(pad, cx, cy))
                    ys.append(0)
                    break
    return np.stack(xs), np.array(ys)


def normalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch.clamp(0, 1) - MEAN) / STD


def augment(batch: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    if random.random() < 0.5:
        t = t.flip(3)
    if random.random() < 0.5:
        t = t.flip(2)
    t = torch.rot90(t, random.randint(0, 3), (2, 3))
    return normalize(t * (1 + 0.15 * (torch.rand(len(t), 1, 1, 1) - 0.5)) + 0.08 * (torch.rand(len(t), 3, 1, 1) - 0.5))


def train(x: np.ndarray, y: np.ndarray, epochs: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-4, total_steps=epochs * math.ceil(len(x) / 64), pct_start=0.15
    )
    weight = torch.tensor([1.0, float((1 - y).sum() / y.sum()) ** 0.5])
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order, total = np.random.permutation(len(x)), 0.0
        for s in range(0, len(x), 64):
            idx = order[s : s + 64]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(augment(x[idx])), torch.from_numpy(y[idx]).long(), weight=weight)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{epochs} loss {total / len(x):.3f} ({time.time() - start:.0f}s)", flush=True)
    return model.eval()


def heatmap(model: nn.Module, image: bytes, width: int, height: int, stride: int):
    pad = padded(image)
    xs, ys = list(range(HALF, width - HALF + 1, stride)), list(range(HALF, height - HALF + 1, stride))
    windows = np.stack([window(pad, cx, cy) for cy in ys for cx in xs])
    with torch.no_grad():
        probs = [
            torch.softmax(
                model(normalize(torch.from_numpy(windows[s : s + 256]).permute(0, 3, 1, 2).float() / 255.0)), 1
            )[:, 1]
            for s in range(0, len(windows), 256)
        ]
    return xs, ys, torch.cat(probs).numpy().reshape(len(ys), len(xs))


def peaks(xs: list[int], ys: list[int], heat: np.ndarray, tau: float) -> list[tuple[float, float, float]]:
    """Local maxima above ``tau``, greedily, no two closer than the match radius (strongest first)."""
    found: list[tuple[float, float, float]] = []
    candidates = sorted(
        ((heat[j, i], xs[i], ys[j]) for j in range(len(ys)) for i in range(len(xs)) if heat[j, i] >= tau), reverse=True
    )
    for score, cx, cy in candidates:
        if all(math.hypot(cx - px, cy - py) > RADIUS for _, px, py in found):
            found.append((score, cx, cy))
    return found


def score(found: list[tuple[float, float, float]], gold: list[list[float]]) -> tuple[bool, float]:
    """(exact set, F1) under one-to-one matching within the radius."""
    if not found:
        return False, 0.0
    cost = np.array([[math.hypot(p[1] - g[0], p[2] - g[1]) for g in gold] for p in found])
    rows, cols = linear_sum_assignment(cost)
    tp = int((cost[rows, cols] <= RADIUS).sum())
    return tp == len(gold) == len(found), 2 * tp / (len(found) + len(gold))


def evaluate(model: nn.Module, root: str, split: str, stride: int) -> list[dict]:
    tasks = load_rows(root, "tasks", split)
    meta = tasks.select_columns(["points", "n_targets", "width", "height"]).to_list()
    pictures = tasks.select_columns(["image"]).cast_column("image", HFImage(decode=False))
    out = []
    for i, m in enumerate(meta):
        xs, ys, heat = heatmap(model, pictures[i]["image"]["bytes"], m["width"], m["height"], stride)
        out.append({"targets": m["n_targets"], "points": m["points"], "xs": xs, "ys": ys, "heat": heat})
    print(f"  {split}: {len(out)} tasks scanned", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", required=True, help="Local dataset directory (or Hub repo id) with fields and tasks."
    )
    parser.add_argument("--stride", type=int, default=24, help="Window stride in pixels.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    root = os.path.expanduser(args.dataset)
    torch.set_num_threads(args.threads)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    started = time.time()
    x, y = training_windows(root, random.Random(1))
    print(
        f"{len(x)} training windows: {int(y.sum())} mitotic figures, {int((1 - y).sum())} imposters and tissue",
        flush=True,
    )
    model = train(x, y, args.epochs)
    validation = evaluate(model, root, "validation", args.stride)
    taus = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    f1s = {t: np.mean([score(peaks(v["xs"], v["ys"], v["heat"], t), v["points"])[1] for v in validation]) for t in taus}
    tau = max(f1s, key=f1s.get)
    print("validation mean F1 by threshold:", {t: round(f, 3) for t, f in f1s.items()}, "-> chosen", tau, flush=True)
    test = evaluate(model, root, "test", args.stride)

    print("\nResNet-18 sliding-window detector on the benchmark test tasks (threshold chosen on validation)")
    print("| images | tasks | exact-set pass | mean F1 |\n| --- | --- | --- | --- |")
    results = [score(peaks(v["xs"], v["ys"], v["heat"], tau), v["points"]) for v in test]
    print(
        f"| benchmark | {len(results)} | {np.mean([r[0] for r in results]):.1%} | {np.mean([r[1] for r in results]):.3f} |"
    )
    oracle = [score(peaks(v["xs"], v["ys"], v["heat"], 0.0)[: v["targets"]], v["points"])[0] for v in test]
    print(f"| benchmark, told how many (oracle) | {len(oracle)} | {np.mean(oracle):.1%} | |")
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
