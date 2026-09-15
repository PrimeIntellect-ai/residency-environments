"""Preserve a verified evaluation, its inputs and recipes, then expose readiness.

Registry reads never execute recipe code. Exported bundles are validated again
before they can be used by the installed environment.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from blobkit.registry import Registry
from physim.bundles import Bundle, digest

ROOT = Path(__file__).resolve().parents[2]


def export_bundle(registry, identifier, destination):
    record = registry.get(identifier, kind="world-record")["payload"]
    payloads = {}
    for name, artifact in record["artifacts"].items():
        if not name.startswith("bundle/"):
            continue
        name = name.removeprefix("bundle/")
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or "\\" in name or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError(f"Unsafe bundle artifact path: {name}")
        payloads[name] = registry.read_artifact(artifact)
    if "manifest.json" not in payloads:
        raise ValueError("World record does not preserve a complete evaluation bundle")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    bundle = Bundle(destination)
    bundle.check_runtime()
    if bundle.references() != record["links"]:
        raise ValueError("Exported bundle does not match the world record")
    return bundle


def register(source, registry_root, name, *, pilot_config=None, source_archives=()):
    source = Path(source).resolve()
    registry = Registry(registry_root)
    bundle = Bundle(source / "bundle")
    bundle.check_runtime()
    for case in bundle.suite["cases"]:
        bundle.truth(case)
    native = json.loads((source / "native_validation.json").read_text())
    reference = json.loads((source / "reference-demo/reference_check.json").read_text())
    if not native.get("ready") or not reference.get("ok"):
        raise ValueError("Native and reference validation must have passed")
    if native["references"] != bundle.references() or reference["references"] != bundle.references():
        raise ValueError("Validation refers to a different evaluation bundle")
    origin = json.loads((source / "preparation/origin.json").read_text())
    upstream = Path(origin["source"])
    if not upstream.is_absolute():
        upstream = ROOT / upstream
    if digest(upstream / "final_state.npz") != origin["source_fields_sha256"]:
        raise ValueError("Original simulation fields changed")

    artifacts = {
        "bundle/" + p.relative_to(bundle.root).as_posix(): p.read_bytes()
        for p in sorted(bundle.root.rglob("*"))
        if p.is_file()
    }
    for relative in (
        "native_validation.json",
        "reference-demo/reference_check.json",
        "control_summary.json",
        "preparation/origin.json",
        "programs.json",
        "pilot_summary.json",
        "causal/result.json",
        "causal_extended/result.json",
    ):
        if (source / relative).is_file():
            artifacts["evidence/" + relative] = (source / relative).read_bytes()
    if (source / "pilot_summary.json").is_file():
        pilot = json.loads((source / "pilot_summary.json").read_text())
        pilot_root = (source / pilot["artifact_path"]).resolve()
        if not pilot_root.is_relative_to(source):
            raise ValueError("Pilot artifact path is outside the world output")
        state = json.loads((pilot_root / "laboratory_state.json").read_text())
        for path in [pilot_root / "grade.json", *sorted((pilot_root / "observations").glob("*.npz"))]:
            artifacts["evidence/pilot/" + path.relative_to(pilot_root).as_posix()] = path.read_bytes()
        for path in sorted(Path(state["artifact"]).rglob("*")):
            if path.is_file():
                artifacts["evidence/pilot/predictor/" + path.relative_to(state["artifact"]).as_posix()] = (
                    path.read_bytes()
                )
        if pilot_config is None:
            raise ValueError("A pilot summary requires its explicit --pilot-config")
        artifacts["evidence/pilot/config.toml"] = Path(pilot_config).read_bytes()
    for p in sorted((source / "science").rglob("*")):
        if p.name in ("receipt.json", "observations.npz"):
            artifacts["evidence/" + p.relative_to(source).as_posix()] = p.read_bytes()

    # Replay can start directly from the archived, hash-checked simulation end
    # state. The upstream initialization protocol and executed sources are also
    # preserved, so the longer generation history remains inspectable.
    sources = {}
    for p in sorted(upstream.glob("*")):
        if p.name in (
            "genome.json",
            "final_state.npz",
            "protocol.json",
            "rotation.json",
            "summary.json",
            "download.json",
        ):
            sources["inputs/original/" + p.name] = registry.artifact(p.read_bytes())
    for filename in (
        "eval_preparation.py",
        "build_evaluation_bundle.py",
        "bf_feedback_check.py",
        "xv_feedback_check.py",
        "register_evaluation.py",
        "validate_evaluation.py",
        "fetch_inputs.py",
        "EVALUATION_WORKFLOW.md",
    ):
        path = ROOT / "generators/physim" / filename
        sources["generators/physim/" + filename] = registry.artifact(path.read_bytes())
    for archive in source_archives:
        archive = Path(archive)
        if not archive.is_dir():
            raise ValueError(f"Source archive does not exist: {archive}")
        for path in sorted(archive.rglob("*")):
            if path.is_file():
                sources["executed/" + digest(path) + "/" + path.name] = registry.artifact(path.read_bytes())
    recipe = registry.put(
        "recipe",
        dict(
            name=name + "_preparation",
            kind="prepared-evaluation",
            sources=sources,
            entrypoint="generators/physim/eval_preparation.py",
            command=f"python generators/physim/eval_preparation.py --world {origin['world']} --source inputs/original --output replay",
            upstream_protocol=sources["inputs/original/protocol.json"],
            implementation=origin["implementation"],
            note="Recipe replay creates new data files; the preserved bundle is the byte-exact evaluation release.",
        ),
        refs=list(sources.values()),
    )
    evidence_ids = {key: registry.artifact(value) for key, value in artifacts.items() if key.startswith("evidence/")}
    run = registry.put(
        "generation-run",
        dict(
            name=name,
            kind="prepared-evaluation",
            recipe=recipe,
            origin=origin,
            programs=len(bundle.suite["cases"]),
            development_members=3,
            grading_members=2,
            evidence=evidence_ids,
            references=bundle.references(),
        ),
        refs=[recipe, *evidence_ids.values()],
    )
    identifier = registry.add_source_world(
        name,
        json.loads((bundle.root / "world.json").read_text()),
        source=dict(
            kind="fresh-simulation-preparation",
            origin=origin,
            scope="One prepared laboratory and its verified evaluation suite.",
        ),
        artifacts=artifacts,
        links=bundle.references(),
        recipe=recipe,
        run=run,
    )
    # Prove that the registry alone can reconstruct the runnable bundle before
    # exposing eval-ready in the mutable browsing projection.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="physim-registry-roundtrip-") as temporary:
        restored = export_bundle(registry, identifier, Path(temporary) / "bundle")
        for case in restored.suite["cases"]:
            restored.truth(case)
    path = registry.root / "availability.json"
    availability = (
        json.loads(path.read_text())
        if path.exists()
        else dict(schema_version="physim-registry-availability-v1", eval_ready={})
    )
    availability["eval_ready"][identifier] = dict(
        references=bundle.references(),
        evidence=list(evidence_ids.values()),
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(availability, indent=2) + "\n")
    temporary.replace(path)
    receipt = dict(
        world_record=identifier, recipe=recipe, run=run, references=bundle.references(), registry_roundtrip=True
    )
    (source / "registry_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add")
    add.add_argument("--source", required=True, type=Path)
    add.add_argument("--name", required=True)
    add.add_argument("--registry", type=Path, default=ROOT / "registry")
    add.add_argument("--pilot-config", type=Path, help="Exact config for an included model pilot")
    add.add_argument(
        "--source-archive", type=Path, action="append", default=[], help="Additional executed sources to archive"
    )
    export = subparsers.add_parser("export")
    export.add_argument("world_record")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--registry", type=Path, default=ROOT / "registry")
    args = parser.parse_args()
    if args.command == "add":
        print(
            json.dumps(
                register(
                    args.source,
                    args.registry,
                    args.name,
                    pilot_config=args.pilot_config,
                    source_archives=args.source_archive,
                ),
                indent=2,
            )
        )
    else:
        print(json.dumps(export_bundle(Registry(args.registry), args.world_record, args.output).references(), indent=2))
