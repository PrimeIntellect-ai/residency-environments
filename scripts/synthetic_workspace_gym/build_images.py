from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1", "PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    return completed.stdout.strip()


def _push(*, name: str, context: Path, dockerfile: Path, public: bool) -> None:
    command = [
        "prime",
        "images",
        "push",
        name,
        "--context",
        str(context),
        "--dockerfile",
        str(dockerfile),
        "--platform",
        "linux/amd64",
        "--public" if public else "--private",
        "--plain",
    ]
    _run(*command)


def _image_inventory() -> list[dict[str, object]]:
    payload = json.loads(_run("prime", "images", "list", "--output", "json", "--num", "100", "--plain"))
    return [dict(item) for item in payload["data"]]


def _resolve_image(inventory: list[dict[str, object]], logical_name: str, tag: str) -> dict[str, str]:
    for image in inventory:
        if image.get("imageName") != logical_name or image.get("imageTag") != tag:
            continue
        if image.get("artifactType") != "CONTAINER_IMAGE" or image.get("status") != "COMPLETED":
            continue
        tagged_reference = image.get("fullImagePath")
        if not isinstance(tagged_reference, str):
            raise RuntimeError(f"Prime image listing did not expose a registry reference: {image}")
        if "@sha256:" in tagged_reference:
            digest_reference = tagged_reference
        else:
            inspection = _run("docker", "buildx", "imagetools", "inspect", tagged_reference)
            match = re.search(r"(?m)^Digest:\s+(sha256:[0-9a-f]{64})\s*$", inspection)
            if match is None:
                raise RuntimeError(f"could not resolve an immutable manifest digest for {tagged_reference}")
            repository = tagged_reference.rsplit(":", 1)[0]
            digest_reference = f"{repository}@{match.group(1)}"
        runtime_reference = image.get("displayRef")
        if not isinstance(runtime_reference, str) or not runtime_reference.startswith("prime/"):
            raise RuntimeError(f"Prime image listing did not expose a runnable public reference: {image}")
        return {"runtime": runtime_reference, "container": digest_reference}
    raise KeyError(f"Prime image was not found after push: {logical_name}:{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the six SWG agent images and trusted grader on Prime.")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--name-prefix", default="synthetic-workspace-gym")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", action="store_true", help="Publish images publicly (private is the default).")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve existing completed tags without submitting new builds.",
    )
    args = parser.parse_args()
    script_root = Path(__file__).parent
    logical_names: dict[str, str] = {}
    for context in sorted((args.artifacts / "images").iterdir()):
        if not context.is_dir():
            continue
        logical_name = f"{args.name_prefix}-{context.name}"
        if not args.resolve_only:
            _push(
                name=f"{logical_name}:{args.tag}",
                context=context,
                dockerfile=script_root / "agent.Dockerfile",
                public=args.public,
            )
        logical_names[context.name] = logical_name
    grader_name = f"{args.name_prefix}-grader"
    if not args.resolve_only:
        _push(
            name=f"{grader_name}:{args.tag}",
            context=script_root,
            dockerfile=script_root / "grader.Dockerfile",
            public=args.public,
        )
    logical_names["grader"] = grader_name

    inventory = _image_inventory()
    images = {key: _resolve_image(inventory, name, args.tag) for key, name in logical_names.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(images, indent=2, sort_keys=True) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
