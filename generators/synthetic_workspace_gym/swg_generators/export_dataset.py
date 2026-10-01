from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from swg_generators.generators.registry import get_generator
from swg_generators.provenance import generation_fingerprint
from swg_generators.secure_protocol import build_protocol
from swg_generators.utils.paths import list_relative_files

DEFAULT_MANIFESTS = (
    "sft-easy-v4",
    "sft-validation-v4",
    "rl-hard-v4",
    "rl-eval-v4",
    "eval-d1-d4-paired-panel-48-v4",
    "eval-d5-family-calibration-40-v4",
)
FORMAT = "swg-hub-and-images-v1"


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            compressed.write(_json_bytes(payload))


def _read_manifest(manifest_dir: Path, name: str) -> dict[str, Any]:
    path = manifest_dir / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != name or not isinstance(payload.get("assignments"), list):
        raise ValueError(f"invalid curriculum manifest: {path}")
    return payload


def _generate_task(
    assignment: dict[str, Any],
    *,
    validate: bool,
    generator_revision: str,
) -> dict[str, Any]:
    generator = get_generator(str(assignment["family"]))
    spec = generator.sample_spec(
        difficulty=int(assignment["difficulty"]),
        seed=int(assignment["seed"]),
        scenario_id=str(assignment["scenario"]),
        generation_params={
            "split": str(assignment["split"]),
            "task_id": str(assignment["task_id"]),
        },
    )
    with tempfile.TemporaryDirectory(prefix="swg-artifact-export-") as temporary_dir:
        generated = generator.generate_instance(spec, Path(temporary_dir), validate=validate)
        manifest = generated.manifest.to_dict()
        evaluator_entrypoint = str(manifest["evaluator_entrypoint"]).replace(
            "swg_generators.evaluators.",
            "synthetic_workspace_gym.evaluators.",
            1,
        )
        manifest["evaluator_entrypoint"] = evaluator_entrypoint
        manifest["metadata"]["release_provenance"]["generation_fingerprint"] = generation_fingerprint(
            generated.visible_root,
            generated.hidden_root,
            evaluator_entrypoint,
        )
        manifest["metadata"]["release_provenance"]["generator_revision"] = generator_revision
        return {
            "manifest": manifest,
            "visible_files": {
                path: (generated.visible_root / path).read_bytes()
                for path in list_relative_files(generated.visible_root)
            },
            "hidden_files": {
                path: base64.b64encode((generated.hidden_root / path).read_bytes()).decode("ascii")
                for path in list_relative_files(generated.hidden_root)
            },
            "validation": (
                dict(generated.validation_result.diagnostics.get("generation_validation", {}))
                if generated.validation_result is not None
                else {}
            ),
        }


def _task_prompt(manifest: dict[str, Any]) -> str:
    descriptor = dict(manifest.get("metadata", {}).get("task_descriptor", {}))
    lines = [str(manifest["instruction"])]
    required_output = next(
        (descriptor[key] for key in ("required_output_path", "output_path", "target_path") if descriptor.get(key)),
        None,
    )
    if required_output:
        lines.extend(["", f"Required final artifact: {required_output}"])
    if descriptor.get("entrypoint"):
        lines.append(f"Visible check/entrypoint: {descriptor['entrypoint']}")
    return "\n".join(lines)


def _blob_path(task_id: str) -> str:
    return f"tasks/{hashlib.sha256(task_id.encode()).hexdigest()}.json.gz"


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def export_artifacts(
    manifest_dir: Path,
    output_dir: Path,
    *,
    manifests: tuple[str, ...] = DEFAULT_MANIFESTS,
    validate: bool = True,
    max_workers: int = 1,
    generator_revision: str,
) -> dict[str, Any]:
    """Export grader-only Hub payloads and per-curriculum visible image contexts."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_root = output_dir / "dataset"
    images_root = output_dir / "images"
    dataset_root.mkdir(parents=True)
    images_root.mkdir(parents=True)

    curricula = {name: _read_manifest(manifest_dir, name) for name in manifests}
    assignments: dict[str, dict[str, Any]] = {}
    for curriculum in curricula.values():
        for raw_assignment in curriculum["assignments"]:
            assignment = dict(raw_assignment)
            task_id = str(assignment["task_id"])
            previous = assignments.setdefault(task_id, assignment)
            if previous != assignment:
                raise ValueError(f"conflicting assignments for task {task_id}")

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if re.fullmatch(r"[0-9a-f]{40}", generator_revision) is None:
        raise ValueError("generator_revision must be a full lowercase Git commit SHA")
    ordered_task_ids = sorted(assignments)
    generate = partial(
        _generate_task,
        validate=validate,
        generator_revision=generator_revision,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        generated_rows = executor.map(generate, (assignments[task_id] for task_id in ordered_task_ids))

    generated_tasks: dict[str, dict[str, Any]] = {}
    for task_id, generated in zip(ordered_task_ids, generated_rows, strict=True):
        execution_plan, execution_files, grader_files = build_protocol(
            generated["manifest"],
            generated["visible_files"],
            generated["hidden_files"],
        )
        generated["execution_plan"] = execution_plan
        generated["execution_files"] = execution_files
        generated["hidden_files"] = grader_files
        generated_tasks[task_id] = generated
        _write_gzip_json(
            dataset_root / _blob_path(task_id),
            {
                "task_id": task_id,
                "manifest": generated["manifest"],
                "execution_plan": generated["execution_plan"],
                "execution_files": generated["execution_files"],
                "hidden_files": generated["hidden_files"],
            },
        )

    counts: dict[str, int] = {}
    for name, curriculum in curricula.items():
        records: list[dict[str, Any]] = []
        workspaces_root = images_root / name / "workspaces"
        for raw_assignment in curriculum["assignments"]:
            assignment = dict(raw_assignment)
            task_id = str(assignment["task_id"])
            generated = generated_tasks[task_id]
            provenance = dict(generated["manifest"].get("metadata", {}).get("release_provenance", {}))
            records.append(
                {
                    **assignment,
                    "prompt": _task_prompt(generated["manifest"]),
                    "generation_fingerprint": str(provenance.get("generation_fingerprint", "")),
                    "task_blob": _blob_path(task_id),
                    "visible_files": sorted(generated["visible_files"]),
                    "initial_digests": {
                        path: hashlib.sha256(content).hexdigest()
                        for path, content in generated["visible_files"].items()
                    },
                }
            )
            _write_tree(workspaces_root / task_id, generated["visible_files"])

        counts[name] = len(records)
        manifest_path = dataset_root / "manifests" / f"{name}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(b"".join(_json_bytes(record) for record in records))
        (images_root / name / "task_ids.json").write_bytes(
            (json.dumps([record["task_id"] for record in records], indent=2) + "\n").encode("utf-8")
        )

    metadata = {
        "format": FORMAT,
        "generator_revision": generator_revision,
        "manifests": list(manifests),
        "num_unique_tasks": len(assignments),
        "manifest_task_counts": counts,
    }
    validation_rows = [
        {
            "task_id": task_id,
            "family": generated["manifest"]["family"],
            "difficulty": generated["manifest"]["difficulty"],
            **generated["validation"],
        }
        for task_id, generated in sorted(generated_tasks.items())
    ]
    measured_untouched = [
        float(row["unmodified_reward"]) for row in validation_rows if row.get("unmodified_reward") is not None
    ]
    (dataset_root / "validation.json").write_bytes(
        (
            json.dumps(
                {
                    "all_reference_solutions_passed": all(
                        float(row.get("reference_solution_score", 0.0)) == 1.0 for row in validation_rows
                    ),
                    "max_unmodified_reward": max(measured_untouched, default=None),
                    "tasks_validated": len(validation_rows),
                    "tasks_with_unmodified_limit": len(measured_untouched),
                    "rows": validation_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    (dataset_root / "metadata.json").write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (output_dir / "integrity.json").write_bytes(
        (
            json.dumps(
                {
                    "format": FORMAT,
                    "dataset_files": _hash_tree(dataset_root),
                    "image_context_files": _hash_tree(images_root),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SWG Hub data and split-isolated image contexts.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("configs/synthetic_workspace_gym"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--generator-revision", required=True)
    args = parser.parse_args()
    metadata = export_artifacts(
        args.manifest_dir,
        args.output_dir,
        validate=not args.no_validate,
        max_workers=args.max_workers,
        generator_revision=args.generator_revision,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
