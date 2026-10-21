# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Upload a built ``midog-mitosis`` dataset directory to the Hugging Face Hub.

The repository is created private unless ``--public`` is given. Authenticate first with ``hf auth login`` or
``HF_TOKEN``. A partial (``--slides``) build is refused: run ``validate_dataset.py`` on the full build first. Remote
shards that the local build no longer has are deleted, and the dataset card's usage snippet is filled in with the
target repository id.

    uv run generators/midog-mitosis/upload_dataset.py --repo-id <user>/midog-mitosis --dataset ~/midog-build/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID_TOKEN = "__REPO_ID__"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Target dataset repository, e.g. <user>/midog-mitosis.")
    parser.add_argument("--dataset", default="~/midog-build/dataset", help="Directory written by build_dataset.py.")
    parser.add_argument("--public", action="store_true", help="Create the repository public (default: private).")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser()
    for required in ("README.md", "build_report.json", "tasks", "warmup", "fields", "candidates"):
        if not (root / required).exists():
            print(f"{root} is missing {required}; run build_dataset.py and validate_dataset.py first", file=sys.stderr)
            return 1
    if json.loads((root / "build_report.json").read_text()).get("partial"):
        print(f"{root} is a partial (--slides) build and must not be published", file=sys.stderr)
        return 1
    card = (root / "README.md").read_text().replace(REPO_ID_TOKEN, args.repo_id)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(root),
        ignore_patterns=["README.md"],
        delete_patterns=["tasks/*", "warmup/*", "fields/*", "candidates/*"],  # a stale shard must not stay behind
        commit_message="Upload midog-mitosis data",
    )
    commit = api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print(f"uploaded to https://huggingface.co/datasets/{args.repo_id} ({'public' if args.public else 'private'})")
    print(commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
