from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def ensure_within_root(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path '{relative_path}' escapes workspace root '{root}'.")
    return candidate


def list_relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    files = [str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()]
    return sorted(files)


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
