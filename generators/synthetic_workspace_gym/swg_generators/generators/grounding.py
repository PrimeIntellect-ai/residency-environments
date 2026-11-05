from __future__ import annotations

import json
from collections.abc import Mapping

from swg_generators.evaluators.metrics import flatten_json

STRUCTURAL_SENTINELS = {"<empty_list>", "<empty_object>"}


def validate_exact_json_grounding(
    expected: object,
    visible_files: Mapping[str, object],
    *,
    target_path: str,
    exempt_paths: set[str] | None = None,
) -> dict[str, list[str]]:
    """Require every scored scalar to have an agent-visible evidence source."""

    exemptions = exempt_paths or set()
    searchable = {path: str(content) for path, content in visible_files.items() if path != target_path}
    evidence: dict[str, list[str]] = {}
    missing: list[str] = []
    for leaf in flatten_json(expected):
        path = str(leaf["path"])
        value = leaf["value"]
        if path in exemptions or value in STRUCTURAL_SENTINELS:
            continue
        needle = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        sources = sorted(path for path, content in searchable.items() if str(needle) in content)
        if sources:
            evidence[path] = sources
        else:
            missing.append(f"{path}={needle!r}")
    if missing:
        raise ValueError("Exact-JSON answer is not grounded in visible evidence: " + "; ".join(missing))
    return evidence
