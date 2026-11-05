from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def scratch_directory(base_dir: Path, prefix: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{prefix}{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
