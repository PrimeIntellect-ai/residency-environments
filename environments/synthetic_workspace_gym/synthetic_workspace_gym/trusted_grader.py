from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def main() -> None:
    if len(sys.argv) != 8:
        raise SystemExit(
            "usage: trusted_grader.py WORKSPACE HIDDEN MANIFEST INITIAL_DIGESTS EXECUTION EXECUTION_OUTPUTS OUTPUT"
        )
    workspace, hidden, manifest_path, initial_path, execution_path, execution_outputs, output_path = map(
        Path,
        sys.argv[1:],
    )
    sys.path.insert(0, str(Path(__file__).parent / "lib"))

    from synthetic_workspace_gym.evaluators.registry import get_evaluator
    from synthetic_workspace_gym.schemas import EnvironmentManifest

    manifest = EnvironmentManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    (hidden / "precomputed_execution.json").write_bytes(execution_path.read_bytes())
    trusted_outputs = hidden / "precomputed_outputs"
    shutil.rmtree(trusted_outputs, ignore_errors=True)
    shutil.copytree(execution_outputs, trusted_outputs)
    current = _file_digests(workspace)
    evaluator = get_evaluator(
        manifest.family,
        evaluator_entrypoint=manifest.evaluator_entrypoint,
    )
    result = evaluator.evaluate(workspace, manifest, hidden).to_dict()
    result["changed_file_count"] = sum(
        1 for path in set(initial) | set(current) if initial.get(path) != current.get(path)
    )
    result["final_file_count"] = len(current)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
