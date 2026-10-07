from __future__ import annotations

import hashlib
from pathlib import Path

ENVIRONMENT_VERSION = "0.1.0"
EVALUATOR_VERSION = "swg-capability-evaluators-v4"


def generation_fingerprint(
    visible_root: Path,
    hidden_root: Path,
    evaluator_entrypoint: str,
) -> str:
    """Hash the complete generated task and evaluator inputs deterministically."""

    digest = hashlib.sha256()
    digest.update(evaluator_entrypoint.encode("utf-8"))
    digest.update(b"\0")
    for label, root in (("visible", visible_root), ("hidden", hidden_root)):
        paths = sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
        for path in paths:
            relative = path.relative_to(root).as_posix()
            digest.update(label.encode("ascii"))
            digest.update(b"/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()
