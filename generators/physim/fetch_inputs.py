"""Restore BF or XV preparation inputs from an explicitly pinned HF registry.

Reads data and archives source bytes without executing any downloaded code.
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from blobkit.registry import Registry
from huggingface_hub import hf_hub_download
from physim.hub import fetch_catalog


def fetch(*, repo, revision, world, output, offline=False):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable 40-character dataset commit")
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new input directory")
    catalog = fetch_catalog(repo=repo, revision=revision, offline=offline)
    matches = [record for record in catalog["worlds"] if record["world_name"] == world]
    if len(matches) != 1:
        raise ValueError(f"Expected one world record named {world}, found {len(matches)}")
    with tempfile.TemporaryDirectory(prefix="physim-inputs-") as temporary:
        registry = Registry(temporary)

        def read(identifier):
            destination = registry.path(identifier)
            remote = hf_hub_download(
                repo,
                "registry/" + destination.relative_to(registry.root).as_posix(),
                repo_type="dataset",
                revision=revision,
                local_files_only=offline,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote, destination)
            return (
                registry.read_artifact(identifier) if identifier.startswith("artifact:") else registry.get(identifier)
            )

        record = read(matches[0]["world_record"])
        recipe_id = record["payload"]["recipe"]
        if recipe_id is None:
            raise ValueError("This record has no archived preparation recipe")
        recipe = read(recipe_id)["payload"]
        names = ("genome.json", "final_state.npz", "protocol.json", "rotation.json", "summary.json")
        artifacts = {
            name: recipe["sources"]["inputs/original/" + name]
            for name in names
            if "inputs/original/" + name in recipe["sources"]
        }
        if not {"genome.json", "final_state.npz", "protocol.json"} <= artifacts.keys():
            raise ValueError("Recipe lacks the saved preparation inputs; use the published bundle for this record")
        data = {name: read(identifier) for name, identifier in artifacts.items()}
        genome = read(record["payload"]["genome"])["payload"]
        if json.loads(data["genome.json"]) != genome:
            raise ValueError("Archived preparation genome differs from its world record")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        (output / name).write_bytes(value)
    receipt = dict(repo=repo, revision=revision, world_record=record["id"], recipe=recipe_id, artifacts=artifacts)
    (output / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--world", required=True, choices=["bf_trail_lab", "xv_rotor_lab"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fetch(**vars(args)), indent=2))
