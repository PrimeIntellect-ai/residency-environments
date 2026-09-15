"""Build a verified catalog projection for the registry and static documentation."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from blobkit.registry import KINDS, Registry

ROOT = Path(__file__).resolve().parents[2]


def build_catalog(root):
    """Distinguish preserved worlds from explicitly prepared, runnable evaluations."""
    registry = Registry(root)
    catalog = registry.catalog()
    path = registry.root / "availability.json"
    availability = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": "physim-registry-availability-v1",
            "eval_ready": {},
        }
    )
    if availability.get("schema_version") != "physim-registry-availability-v1":
        raise ValueError("Unsupported Physim availability schema")
    worlds = {world["id"]: world for world in catalog["worlds"]}
    entries = availability["eval_ready"]
    if set(entries) - worlds.keys():
        raise ValueError("Availability references unknown world records")
    for identifier, world in worlds.items():
        world["status"] = "preserved"
        if identifier not in entries:
            continue
        evaluation = entries[identifier]
        references = evaluation["references"]
        if any(not references.get(key) for key in ("world", "preparation", "suite", "bundle")):
            raise ValueError("Eval-ready worlds require world, preparation, suite and bundle identities")
        if references != world.get("links"):
            raise ValueError("Evaluation identities do not match the preserved world record")
        if not evaluation["evidence"]:
            raise ValueError("Eval-ready worlds require validation evidence")
        for artifact in evaluation["evidence"]:
            registry.read_artifact(artifact)
        world["status"] = "eval-ready"
        world["evaluation"] = evaluation
    catalog["availability"] = {
        "schema_version": availability["schema_version"],
        "counts": dict(sorted(Counter(w["status"] for w in worlds.values()).items())),
    }
    return catalog


def export_catalog(root, sync_docs=False):
    registry = Registry(root)
    catalog = build_catalog(root)
    data = json.dumps(catalog, indent=2, allow_nan=False) + "\n"
    (registry.root / "index.json").write_text(data)
    if sync_docs:
        (ROOT / "docs_source/registry.json").write_text(data)
    return catalog["verification"]


def stage_registry(root, destination):
    """Copy verified records and evaluation availability into a fresh release."""
    root, destination = Path(root), Path(destination)
    catalog = build_catalog(root)
    destination.mkdir(parents=True, exist_ok=False)
    for kind in sorted(KINDS | {"artifact"}):
        suffix = "*.bin" if kind == "artifact" else "*.json"
        for source in sorted((root / kind).glob(suffix)):
            target = destination / kind / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if (root / "availability.json").exists():
        shutil.copyfile(root / "availability.json", destination / "availability.json")
    if build_catalog(destination) != catalog:
        raise ValueError("Staged registry differs from its source")
    (destination / "index.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "registry")
    parser.add_argument("--sync-docs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_catalog(args.registry, args.sync_docs), indent=2))
