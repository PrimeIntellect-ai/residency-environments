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
"""Mutation test for ``validate_dataset.py``: corrupt a copy of the test tasks and require every fault to be caught.

    uv run generators/midog-mitosis/mutation_test.py --work-dir ~/midog-build

A validator that passes everything proves nothing, so this plants nine faults in ``<work-dir>/mutant`` (a copy of the
built dataset whose test tasks are altered), runs ``validate_dataset.py`` on it without the slow pixel spot-check, and
exits non-zero unless the validator reports each fault.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

from datasets import Dataset
from datasets import Image as HFImage
from PIL import Image


def plant(rows: list[dict]) -> list[tuple[str, str]]:
    """Corrupt ``rows`` in place; returns ``(fault, pattern)`` where the pattern must appear in the validator's output."""
    singles = [r for r in rows if len(r["parts"]) == 1]
    pairs = [r for r in rows if len(r["parts"]) == 2]
    multi = [r for r in singles if r["n_targets"] >= 2]
    faults = []

    def fault(name: str, row: dict, pattern: str | None = None) -> None:
        faults.append((name, re.escape(row["task_id"]) if pattern is None else pattern))

    row = multi[0]
    row["answer"] = re.sub(r"\d+\.\d", lambda m: f"{float(m.group()) + 60:.1f}", row["answer"], count=1)
    fault("answer: a coordinate off by 60 normalized units", row)

    row = multi[1]
    row["points"][0][0] += 5.0
    fault("points: a reference center shifted by 5 px", row)

    row = multi[2]
    row["n_targets"] += 1
    fault("n_targets: one more than the figures listed", row)

    a, b = singles[0], singles[1]
    a["image"], b["image"] = b["image"], a["image"]
    fault("image: swapped with another task's", a)

    row = pairs[0]
    whole = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    half = whole.width // 2
    swapped = Image.new("RGB", whole.size)
    swapped.paste(whole.crop((half, 0, whole.width, whole.height)), (0, 0))
    swapped.paste(whole.crop((0, 0, half, whole.height)), (half, 0))
    buffer = io.BytesIO()
    swapped.save(buffer, format="JPEG", quality=92)
    row["image"] = {"bytes": buffer.getvalue(), "path": None}
    fault("image: the two halves of a pair swapped", row)

    first = pairs[1]
    other = next(p for p in pairs[2:] if p["slide"] != first["slide"])
    first["parts"] = [first["parts"][0], other["parts"][1]]
    fault("parts: a pair joining two slides", first)

    row = pairs[3]
    row["parts"] = [row["parts"][0], row["parts"][0]]
    fault("parts: a pair of one field twice", row)

    rows[5]["rank"] = rows[6]["rank"]
    faults.append(("rank: duplicated", "ranks are not 0..n-1"))

    rows.remove(singles[10])
    faults.append(("a one-field task removed", "expected 512 tasks, got 511"))
    return faults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default="~/midog-build")
    args = parser.parse_args()
    work = Path(args.work_dir).expanduser()
    source, mutant = work / "dataset", work / "mutant"
    shutil.rmtree(mutant, ignore_errors=True)
    (mutant / "tasks").mkdir(parents=True)
    for name in ("fields", "candidates", "warmup"):
        (mutant / name).symlink_to(source / name)
    for name in ("README.md", "build_report.json"):
        shutil.copy(source / name, mutant / name)
    for shard in (source / "tasks").glob("*.parquet"):
        if not shard.name.startswith("test-"):
            (mutant / "tasks" / shard.name).symlink_to(shard)

    test = Dataset.from_parquet(sorted(str(f) for f in (source / "tasks").glob("test-*.parquet")))
    features = test.features
    rows = test.cast_column("image", HFImage(decode=False)).to_list()
    faults = plant(rows)
    Dataset.from_list(rows, features=features).to_parquet(mutant / "tasks" / "test-00000-of-00001.parquet")

    validator = Path(__file__).with_name("validate_dataset.py")
    run = subprocess.run(
        [sys.executable, str(validator), "--work-dir", str(work), "--dataset", str(mutant), "--spot-check", "0"],
        capture_output=True,
        text=True,
    )
    shutil.rmtree(mutant, ignore_errors=True)
    if "PROBLEMS" not in run.stdout:
        print(f"the validator did not finish its checks (exit code {run.returncode}); its last output:")
        print((run.stderr or run.stdout)[-1500:])
        return 1
    missed = 0
    for name, pattern in faults:
        caught = re.search(pattern, run.stdout) is not None
        missed += not caught
        print(f"{'caught' if caught else 'MISSED':7s} {name}")
    print(f"\nvalidator exit code {run.returncode}; {len(faults) - missed} of {len(faults)} faults caught")
    return int(missed > 0 or run.returncode == 0)


if __name__ == "__main__":
    sys.exit(main())
