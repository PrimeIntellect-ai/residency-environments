from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def publish_dataset(
    dataset_dir: Path,
    repo_id: str,
    *,
    private: bool,
    revision: str,
) -> str:
    if not (dataset_dir / "metadata.json").is_file():
        raise ValueError(f"not an SWG dataset export: {dataset_dir}")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message="Publish immutable Synthetic Workspace Gym artifacts",
    )
    if not commit.oid:
        raise RuntimeError("Hugging Face did not return a commit revision")
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish an SWG dataset export to Hugging Face.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Publish the grader-bearing dataset publicly (private is the default).",
    )
    args = parser.parse_args()
    revision = publish_dataset(
        args.dataset_dir,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
    )
    print(json.dumps({"repo_id": args.repo_id, "revision": revision}, indent=2))


if __name__ == "__main__":
    main()
