from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def convert_to_json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return {key: convert_to_json_compatible(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): convert_to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [convert_to_json_compatible(item) for item in value]
    return value


def to_json_compatible(value: Any) -> Any:
    """Convert a value into a JSON-safe structure."""

    return convert_to_json_compatible(value)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(to_json_compatible(payload), indent=indent, sort_keys=True).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join((json.dumps(to_json_compatible(row), sort_keys=True) + "\n").encode("utf-8") for row in rows)
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))
