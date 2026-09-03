"""Package the public, frozen example table as train/test Hugging Face Parquet files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

SOURCE_SHA256 = "9f1cedc98d93f84069d4ed823325adb6e03fca26c611ca42e11628d057b9c048"
EXPECTED_ROWS = {"train": 186854, "test": 20257}
EXPECTED_COLUMNS = {
    "compound",
    "cell_line",
    "entry_point",
    "phenotype",
    "user_prompt",
    "answer_json",
    "split",
    "lincs_dose_um",
    "lincs_time_h",
    "prism_dose_um",
    "prism_doses_um",
    "prism_duration_h",
    "prism_screens",
    "prism_n_full_ids",
    "prism_aggregation",
}
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Source does not match the frozen public dataset")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    frame = pd.read_parquet(args.source)
    if set(frame.columns) != EXPECTED_COLUMNS or frame.groupby("split").size().to_dict() != EXPECTED_ROWS:
        raise ValueError("Unexpected dataset schema or split counts")
    if set(frame[frame.split == "train"].compound) & set(frame[frame.split == "test"].compound):
        raise ValueError("Compounds overlap across train/test")
    for row in frame.itertuples(index=False):
        if not isinstance(row.user_prompt, str) or not isinstance(json.loads(row.answer_json), dict):
            raise ValueError("Invalid prompt/reference row")
    (args.output / "data").mkdir(parents=True)
    manifest = {"source_sha256": SOURCE_SHA256, "rows": len(frame), "files": {}}
    for split, count in EXPECTED_ROWS.items():
        selected = frame[frame.split == split].reset_index(drop=True)
        path = args.output / "data" / f"{split}.parquet"
        selected.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(selected, pd.read_parquet(path))
        manifest["files"][f"data/{split}.parquet"] = {"rows": count, "sha256": sha256(path)}
    shutil.copyfile(ROOT / "scripts/drug-perturbation/DATASET_CARD.md", args.output / "README.md")
    shutil.copyfile(ROOT / "environments/drug-perturbation/DATA_SOURCES.md", args.output / "DATA_SOURCES.md")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
