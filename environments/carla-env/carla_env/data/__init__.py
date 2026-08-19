from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_json", "load_trolley_micro_benchmarks"]

_DATA_DIR = Path(__file__).parent


def load_json(name: str) -> Dict[str, Any]:
    path = _DATA_DIR / name
    with open(path, "r") as f:
        return json.load(f)


def load_trolley_micro_benchmarks() -> Dict[str, Any]:
    return load_json("trolley_micro_benchmarks.json")
