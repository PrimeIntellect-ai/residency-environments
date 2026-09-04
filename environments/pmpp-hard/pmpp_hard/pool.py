"""Load release metadata and construct the configured task pool.

Manifest validation reports line-specific errors, and task construction preserves
the source order and prompt-suffix precedence.
"""

import dataclasses
import hashlib
import json
import pathlib

from pmpp_hard import markers
from pmpp_hard.config import GATE_PERF, PMPPHardConfig, PMPPHardTaskData
from pmpp_hard.errors import PoolIntegrityError
from pmpp_hard.paths import DataTree

_GATE_TYPES = ("perf", "insight", "correctness", "precision", "bit_exactness")
_REQUIRED = ("task_id", "student_file", "prompt", "gate_type", "runtime_kind")
_RUNTIME_KINDS = ("cuda", "triton", "cutlass")

SANITY_SCHEMA_VERSION = 2


def resolve_agent_image(config: PMPPHardConfig, kind: str) -> str | None:
    """Resolve the agent sandbox image for a task's runtime lane."""
    return (config.agent_images or {}).get(kind, config.agent_image)


def load_manifest(path: pathlib.Path) -> list[dict]:
    """Parse and validate a JSONL manifest while preserving additional fields."""
    if not path.exists():
        raise PoolIntegrityError(f"manifest not found: {path}")
    rows: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(path.open(), 1):
        if not line.strip():
            raise PoolIntegrityError(f"manifest {path.name} line {i}: blank line")
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise PoolIntegrityError(
                f"manifest {path.name} line {i}: invalid JSON — {e}"
            ) from e
        for k in _REQUIRED:
            if not isinstance(r.get(k), str) or not r[k]:
                raise PoolIntegrityError(
                    f"manifest {path.name} line {i}: missing/empty field {k!r}"
                )
        if r["gate_type"] not in _GATE_TYPES:
            raise PoolIntegrityError(
                f"manifest {path.name} line {i}: bad gate_type {r['gate_type']!r}"
            )
        if r["runtime_kind"] not in _RUNTIME_KINDS:
            raise PoolIntegrityError(
                f"manifest {path.name} line {i}: bad runtime_kind {r['runtime_kind']!r}"
            )
        if r["task_id"] in seen:
            raise PoolIntegrityError(
                f"manifest {path.name} line {i}: duplicate task_id {r['task_id']!r}"
            )
        seen.add(r["task_id"])
        rows.append(r)
    return rows


@dataclasses.dataclass(frozen=True)
class SanityManifest:
    """Normalized task-validity metadata from the sanity manifest."""

    schema_version: int
    tasks: dict[str, dict]
    path_existed: bool = True

    def valid(self, task_id: str) -> bool:
        e = self.tasks.get(task_id)
        return isinstance(e, dict) and bool(e.get("valid"))


def load_sanity_manifest(path: pathlib.Path) -> SanityManifest:
    """Load sanity metadata for later coverage validation by preflight checks."""
    if not path.exists():
        return SanityManifest(schema_version=0, tasks={}, path_existed=False)
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PoolIntegrityError(
            f"sanity manifest {path.name}: invalid JSON — {e}"
        ) from e
    if not isinstance(d, dict):
        raise PoolIntegrityError(
            f"sanity manifest {path.name}: expected object, got {type(d).__name__}"
        )
    if not isinstance(d.get("tasks"), dict):
        raise PoolIntegrityError(
            f"sanity manifest {path.name}: expected schema-2 object with a tasks map"
        )
    tasks = d["tasks"]
    for tid, entry in tasks.items():
        if not isinstance(entry, dict):
            raise PoolIntegrityError(
                f"sanity manifest {path.name}: entry {tid!r} is {type(entry).__name__}, expected object "
                f"(a flat/corrupted shape) — refusing a silently-degraded manifest"
            )
    try:
        schema = int(d.get("schema"))
    except (TypeError, ValueError) as e:
        raise PoolIntegrityError(
            f"sanity manifest {path.name}: non-numeric schema {d.get('schema')!r}"
        ) from e
    if schema != SANITY_SCHEMA_VERSION:
        raise PoolIntegrityError(
            f"sanity manifest {path.name}: schema {schema}, expected {SANITY_SCHEMA_VERSION}"
        )
    return SanityManifest(schema_version=schema, tasks=tasks)


@dataclasses.dataclass
class PoolStats:
    """Task-pool loading statistics used by preflight validation."""

    rows: list[dict]
    skipped_by_filter: list[str]
    sanity: SanityManifest
    # Requested task identifiers that are absent from the manifest.
    unmatched_include_ids: list[str] = dataclasses.field(default_factory=list)


def build_tasks(
    config: PMPPHardConfig, tree: DataTree
) -> tuple[list[PMPPHardTaskData], PoolStats]:
    """Build tasks in manifest order after applying filters and prompt options."""
    sanity = load_sanity_manifest(tree.sanity_manifest(config.sanity_manifest))
    rows = load_manifest(tree.manifest(config.manifest))
    tasks: list[PMPPHardTaskData] = []
    skipped_filter: list[str] = []
    idx = 0
    for r in rows:
        tid = r["task_id"]
        if config.include_ids and tid not in config.include_ids:
            skipped_filter.append(tid)
            continue
        perf_gated = r["gate_type"] in GATE_PERF and config.require_perf
        prompt = r["prompt"]
        if config.single_shot:
            prompt = prompt + markers.SINGLE_SHOT_INSTRUCTION
        else:
            # Add agentic instructions before any performance feedback guidance.
            if getattr(config, "always_submit", True):
                prompt = prompt + markers.ALWAYS_SUBMIT_INSTRUCTION.format(
                    student_file=r["student_file"]
                )
            if config.held_out and perf_gated and sanity.valid(tid):
                feedback = (
                    markers.TRITON_FEEDBACK_INSTRUCTION
                    if r.get("runtime_kind") == "triton"
                    else markers.FEEDBACK_INSTRUCTION
                )
                prompt = prompt + feedback
        tasks.append(
            PMPPHardTaskData(
                idx=idx,
                name=tid,
                image=resolve_agent_image(config, r["runtime_kind"]),
                prompt=prompt,
                network_allow=[],
                task_id=tid,
                student_file=r["student_file"],
                gate_type=r["gate_type"],
                test_target=r.get("test_target", "test_student"),
                bench_target=r.get("bench_target", "bench_student"),
                perf_gated=perf_gated,
                sanity_ok=sanity.valid(tid),
                perf_ratio_target=r.get("perf_ratio_target"),
                category=r.get("category"),
                family=r.get("family"),
                runtime_kind=r.get("runtime_kind"),
            )
        )
        idx += 1
    manifest_ids = {r["task_id"] for r in rows}
    unmatched_ids = [i for i in (config.include_ids or []) if i not in manifest_ids]
    return tasks, PoolStats(
        rows=rows,
        skipped_by_filter=skipped_filter,
        sanity=sanity,
        unmatched_include_ids=unmatched_ids,
    )


def bundle_sha256(tree: DataTree, task_id: str) -> str:
    """Hash a task bundle from sorted paths and bytes, excluding Python caches."""
    h = hashlib.sha256()
    b = tree.bundle(task_id)
    for f in sorted(b.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or f.suffix == ".pyc":
            continue
        h.update(str(f.relative_to(b)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def pool_content_hash(
    tree: DataTree,
    tasks: list[PMPPHardTaskData],
    rows: list[dict],
    bundle_hashes: dict[str, str] | None = None,
) -> str:
    """Build a stable pool identity from task metadata and bundle content."""
    by_id = {r["task_id"]: r for r in rows}
    bundle_hashes = bundle_hashes or {}
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda t: t.task_id):
        h.update(t.task_id.encode())
        h.update(bundle_hashes.get(t.task_id, bundle_sha256(tree, t.task_id)).encode())
        h.update(json.dumps(by_id.get(t.task_id, {}), sort_keys=True).encode())
    return h.hexdigest()
