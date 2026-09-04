"""Preflight validation for packaged task pools.

Checks use a read-only ``PreflightContext`` and validate the configured release
manifest with its supporting data.

``preflight_waive`` demotes named checks from ERROR to WARN. Disabling
``preflight`` skips the complete suite.
"""

import dataclasses
import difflib
import json
import re
import shutil
import subprocess
from typing import Callable, Literal

from pmpp_hard import grader_inputs, markers
from pmpp_hard.config import LeverVector, PMPPHardConfig, PMPPHardTaskData
from pmpp_hard.errors import PoolIntegrityError
from pmpp_hard.paths import REF_NAMES, DataTree
from pmpp_hard.pool import PoolStats
from pmpp_hard.provision import _DENY_PREFIXES, grader_files, perf_enabled
from pmpp_hard.scoring import (
    modal_gpu_arch_error,
    modal_image_is_registry_addressable,
    modal_prerequisite_error,
    resolve_score_image,
    runtime_kind_of,
    score_image_is_pullable,
)

Severity = Literal["ERROR", "WARN", "INFO"]


class _SidecarError(Exception):
    """Raised when a preflight sidecar is invalid.

    The preflight runner converts this error into a standard finding so configured
    waiver behavior remains available.
    """


def _load_sidecar(path, check: str, *, want: type = dict):
    """Load a JSON sidecar and validate its top-level type."""
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise _SidecarError(f"{path.name}: invalid JSON — {e}") from e
    if not isinstance(d, want):
        raise _SidecarError(
            f"{path.name}: expected {want.__name__}, got {type(d).__name__}"
        )
    return d


_FNV_TOKENS = ("1469598103934665603", "14650fb0739d0383")
_BANNED_AGENT_TOKENS = (
    "expected_hard_mechanism",
    "primary_category",
    "// Calibrated:",
    "PMPP_CANARY_",
)


@dataclasses.dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    task_id: str | None
    message: str


@dataclasses.dataclass(frozen=True)
class PreflightContext:
    tree: DataTree
    config: PMPPHardConfig
    tasks: list[PMPPHardTaskData]
    stats: PoolStats
    pool_hash: str
    missing_starters: frozenset[str] = frozenset()


def _read(p) -> str:
    return p.read_text(errors="replace")


def _per_task(ctx: PreflightContext):
    """Return tasks eligible for checks that require a starter bundle."""
    return [t for t in ctx.tasks if t.task_id not in ctx.missing_starters]


# Core package checks


def check_starter_exists(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in ctx.tasks:
        if not (ctx.tree.bundle(t.task_id) / t.student_file).exists():
            out.append(
                Finding(
                    "starter_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: missing starter {t.student_file}",
                )
            )
    return out


def check_makefile_exists(ctx: PreflightContext) -> list[Finding]:
    return [
        Finding("makefile_exists", "ERROR", t.task_id, f"{t.task_id}: missing Makefile")
        for t in _per_task(ctx)
        if not (ctx.tree.bundle(t.task_id) / "Makefile").exists()
    ]


def check_sanity_overclaim(ctx: PreflightContext) -> list[Finding]:
    return [
        Finding(
            "sanity_overclaim",
            "ERROR",
            t.task_id,
            f"{t.task_id}: sanity manifest claims valid but data/sanity/"
            f"{t.task_id}/ does not exist — manifest over-claims",
        )
        for t in _per_task(ctx)
        if t.sanity_ok and not ctx.tree.sanity_dir(t.task_id).is_dir()
    ]


def check_stub_leak(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        bundle = ctx.tree.bundle(t.task_id)
        stub = _read(bundle / t.student_file)
        for name in REF_NAMES:
            p = bundle / name
            if not p.exists():
                continue
            sim = difflib.SequenceMatcher(None, stub, _read(p)).ratio()
            if sim > 0.5:
                out.append(
                    Finding(
                        "stub_leak",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: starter {t.student_file} resembles {name} "
                        f"(sim={sim:.2f}) — reference leak",
                    )
                )
    return out


def check_canary_integrity(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        m = re.match(r"v2-pl-(\d+)-", t.task_id)
        if not m:
            continue
        bundle = ctx.tree.bundle(t.task_id)
        ref_texts = {
            p.name: _read(p) for p in (bundle / n for n in REF_NAMES) if p.exists()
        }
        if not ref_texts:
            continue
        tok = f"PMPP_CANARY_{m.group(1)}_"
        if not any(tok in text for text in ref_texts.values()):
            out.append(
                Finding(
                    "canary_integrity",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: no {tok}* canary in any held-out reference",
                )
            )
        for vp in (bundle / t.student_file, bundle / "README.md"):
            if vp.exists() and tok in _read(vp):
                out.append(
                    Finding(
                        "canary_integrity",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: canary {tok}* leaked into agent-visible {vp.name}",
                    )
                )
    return out


def check_paired_bench_format(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        if not perf_enabled(ctx.config, t):
            continue
        benches = sorted(ctx.tree.bundle(t.task_id).glob("bench_*"))
        cu = [b for b in benches if b.is_file() and b.suffix in (".cu", ".py")]
        if benches and not any("avg_ms=" in _read(b) for b in cu):
            out.append(
                Finding(
                    "paired_bench_format",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: paired-ratio perf but no bench_* prints 'avg_ms=' "
                    f"— the gate would silently no-op",
                )
            )
        # Output digests connect benchmark timing to the computed outputs. The digest
        # literal may come from the shared helper, so using that helper counts as the signal.
        emits = any(
            tok in _read(b)
            for b in cu
            for tok in ("out_fnv=", "pmpp_bench_digest", "OutFnv")
        )
        if cu and not emits:
            out.append(
                Finding(
                    "bench_out_fnv",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: perf bench emits no out_fnv — timed kernel outputs are "
                    f"unverified (no-op/shape-detect hack possible; see REWARD_HACK_AUDIT)",
                )
            )
    return out


def check_paired_ref_exists(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        if not perf_enabled(ctx.config, t):
            continue
        references = sorted(
            ctx.tree.reference_dir(t.task_id).glob("reference_solution.*")
        )
        if not references:
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: paired-ratio perf but no data/reference/{t.task_id}/"
                    f"reference_solution.* — perf would silently pass",
                )
            )
            continue
        if len(references) != 1:
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: multiple data/reference reference_solution.* "
                    f"variants — scoring selection would be ambiguous",
                )
            )
            continue
        refdir_ref = references[0]
        bundle_references = sorted(
            ctx.tree.bundle(t.task_id).glob("reference_solution.*")
        )
        if not bundle_references:
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: bundle has no reference_solution.* source — "
                    "sync and scoring smoke cannot rebuild the authoritative copy",
                )
            )
            continue
        if len(bundle_references) != 1:
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: bundle has multiple reference_solution.* variants",
                )
            )
            continue
        bundle_ref = bundle_references[0]
        if bundle_ref.name != refdir_ref.name:
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: bundle/reference solution variants differ "
                    f"({bundle_ref.name} vs {refdir_ref.name})",
                )
            )
        elif bundle_ref.read_bytes() != refdir_ref.read_bytes():
            out.append(
                Finding(
                    "paired_ref_exists",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: data/reference copy differs from bundle "
                    f"reference — stale sync",
                )
            )
    return out


# Supporting integrity checks


def check_makefile_targets(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        mk = ctx.tree.bundle(t.task_id) / "Makefile"
        if not mk.exists():
            continue  # Reported by check_makefile_exists.
        targets = set(re.findall(r"^([A-Za-z_][\w.-]*)\s*:", _read(mk), re.M))
        required = (
            {t.test_target, "test_reference"}
            if t.bench_target == ""
            else {t.test_target, t.bench_target, "test_reference", "bench_reference"}
        )
        missing = required - targets
        if missing:
            out.append(
                Finding(
                    "makefile_targets",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: Makefile missing target(s) {sorted(missing)} — "
                    f"grading would fail at make time",
                )
            )
    return out


def check_bench_target_consistency(ctx: PreflightContext) -> list[Finding]:
    out = []
    for t in _per_task(ctx):
        if not perf_enabled(ctx.config, t):
            continue
        if t.bench_target == "":
            out.append(
                Finding(
                    "bench_target_consistency",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: perf-gated but bench_target is empty — "
                    f"correctness-only is the only legal state for bench-less bundles",
                )
            )
        elif not list(ctx.tree.bundle(t.task_id).glob("bench_*")):
            out.append(
                Finding(
                    "bench_target_consistency",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: perf-gated but bundle has no bench_* source",
                )
            )
    return out


def check_student_file_consistency(ctx: PreflightContext) -> list[Finding]:
    out = []
    by_id = {r["task_id"]: r for r in ctx.stats.rows}
    for t in _per_task(ctx):
        kind = by_id.get(t.task_id, {}).get("runtime_kind")
        if kind is None:
            continue
        if (kind == "triton") != t.student_file.endswith(".py"):
            out.append(
                Finding(
                    "student_file_consistency",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: runtime_kind={kind!r} inconsistent with "
                    f"student_file {t.student_file!r}",
                )
            )
    return out


def check_sanity_leak(ctx: PreflightContext) -> list[Finding]:
    """Reject sanity graders that duplicate bundled held-out graders byte for byte."""
    out = []
    for t in _per_task(ctx):
        sd = ctx.tree.sanity_dir(t.task_id)
        if not sd.is_dir():
            continue
        for f in sd.iterdir():
            if not (f.is_file() and f.name.startswith(("test_", "bench_"))):
                continue
            bf = ctx.tree.bundle(t.task_id) / f.name
            if bf.exists() and bf.read_bytes() == f.read_bytes():
                out.append(
                    Finding(
                        "sanity_leak",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: sanity {f.name} is byte-identical to the real "
                        f"grader — held-out copy leaks the real test",
                    )
                )
    return out


def check_sanity_stale(ctx: PreflightContext) -> list[Finding]:
    """Verify files covered by the sanity manifest's content hashes.

    Entries without file hashes are reported because their freshness cannot be
    verified.
    """
    import hashlib

    out = []
    hashless_entries = 0
    for t in _per_task(ctx):
        entry = ctx.stats.sanity.tasks.get(t.task_id) or {}
        if not entry.get("valid"):
            continue
        files = entry.get("files")
        if not isinstance(files, dict):
            hashless_entries += 1
            continue
        sd = ctx.tree.sanity_dir(t.task_id)
        bundle = ctx.tree.bundle(t.task_id)
        shared = {
            path.name
            for path in bundle.iterdir()
            if path.is_file()
            and (path.suffix in (".h", ".hpp", ".cuh") or path.name == "reference.py")
        }
        for name in sorted(shared - files.keys()):
            out.append(
                Finding(
                    "sanity_stale",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: valid sanity entry omits shared grader dependency "
                    f"{name} — local feedback differs from the scorer",
                )
            )
        for name, rec in files.items():
            sp = sd / name
            if not sp.exists():
                out.append(
                    Finding(
                        "sanity_stale",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: sanity manifest lists {name} but the file is gone",
                    )
                )
                continue
            if hashlib.sha256(sp.read_bytes()).hexdigest() != rec.get("sha256"):
                out.append(
                    Finding(
                        "sanity_stale",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: sanity {name} drifted since validation "
                        f"(sha256 mismatch) — self-check would lie vs scorer",
                    )
                )
            src = bundle / name
            if (
                src.exists()
                and rec.get("source_sha256")
                and hashlib.sha256(src.read_bytes()).hexdigest() != rec["source_sha256"]
            ):
                out.append(
                    Finding(
                        "sanity_stale",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: bundle {name} changed since sanity validation "
                        f"(source_sha256 mismatch) — sanity variant is stale",
                    )
                )
    if hashless_entries:
        out.append(
            Finding(
                "sanity_stale",
                "ERROR",
                None,
                f"{hashless_entries} valid sanity entries have no content hashes — "
                f"release69 requires schema-2 hash binding",
            )
        )
    return out


def check_sanity_ref_leak(ctx: PreflightContext) -> list[Finding]:
    """Reject answer-key files in the packaged sanity tree.

    The filename denylist covers reference solutions and answer-bearing oracle
    variants. The contractual ``reference.py`` used by Triton tests remains allowed.
    """
    out = []
    root = ctx.tree.root / "sanity"
    if not root.is_dir():
        return out
    for tid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(tid_dir.iterdir()):
            if f.is_file() and f.name.startswith(_DENY_PREFIXES):
                out.append(
                    Finding(
                        "sanity_ref_leak",
                        "ERROR",
                        tid_dir.name,
                        f"{tid_dir.name}: answer-key file {f.name} present under "
                        f"data/sanity — the reference must never be staged in the "
                        f"shipped held-out tree (delete it)",
                    )
                )
    return out


def check_effective_pool(ctx: PreflightContext) -> list[Finding]:
    """Report unmatched include filters and reject an empty effective task pool.

    This prevents a misspelled filter from producing a zero-task evaluation.
    """
    out = []
    for i in ctx.stats.unmatched_include_ids:
        out.append(
            Finding(
                "effective_pool",
                "WARN",
                None,
                f"include_ids entry {i!r} matched no row in {ctx.config.manifest}",
            )
        )
    if not ctx.tasks:
        detail = ""
        if ctx.stats.unmatched_include_ids:
            detail = f" (unmatched include_ids={ctx.stats.unmatched_include_ids})"
        out.append(
            Finding(
                "effective_pool",
                "ERROR",
                None,
                f"effective task pool is EMPTY for {ctx.config.manifest} — the eval would "
                f"spin up infrastructure and report over 0 tasks{detail}",
            )
        )
    return out


def check_self_check_coverage(ctx: PreflightContext) -> list[Finding]:
    """Report self-check coverage and enforce it when required by configuration."""
    n = sum(
        1 for t in ctx.tasks if t.sanity_ok and ctx.tree.sanity_dir(t.task_id).is_dir()
    )
    out = [
        Finding(
            "self_check_coverage",
            "INFO",
            None,
            f"{n}/{len(ctx.tasks)} tasks self-checkable (validated held-out grader)",
        )
    ]
    if (
        ctx.config.held_out
        and not ctx.config.single_shot
        and not ctx.stats.sanity.path_existed
    ):
        out.append(
            Finding(
                "self_check_coverage",
                "WARN",
                None,
                f"sanity manifest {ctx.config.sanity_manifest} missing — every task "
                f"ships opaque (no agent-side grader)",
            )
        )
    if ctx.config.require_self_check and ctx.config.held_out and n == 0:
        out.append(
            Finding(
                "self_check_coverage",
                "ERROR",
                None,
                f"require_self_check=true but 0/{len(ctx.tasks)} tasks are "
                f"self-checkable — every task would ship opaque",
            )
        )
    return out


def check_perf_census(ctx: PreflightContext) -> list[Finding]:
    """Report performance-gate coverage and validate an expected count when set."""
    n = sum(1 for t in ctx.tasks if perf_enabled(ctx.config, t))
    out = [
        Finding(
            "perf_census",
            "INFO",
            None,
            f"{n}/{len(ctx.tasks)} tasks perf-gated (effective)",
        )
    ]
    if (
        ctx.config.expected_perf_gated is not None
        and n != ctx.config.expected_perf_gated
    ):
        out.append(
            Finding(
                "perf_census",
                "ERROR",
                None,
                f"expected_perf_gated={ctx.config.expected_perf_gated} but effective "
                f"perf-gated count is {n}",
            )
        )
    return out


def check_answer_key_lint(ctx: PreflightContext) -> list[Finding]:
    """Reject authoring metadata and canary tokens in participant-visible content."""
    out = []
    by_id = {r["task_id"]: r for r in ctx.stats.rows}
    for t in _per_task(ctx):
        bundle = ctx.tree.bundle(t.task_id)
        visible = {
            "prompt": by_id.get(t.task_id, {}).get("prompt", ""),
            t.student_file: _read(bundle / t.student_file),
        }
        if (bundle / "README.md").exists():
            visible["README.md"] = _read(bundle / "README.md")
        for tok in _BANNED_AGENT_TOKENS:
            for where, text in visible.items():
                if tok in text:
                    out.append(
                        Finding(
                            "answer_key_lint",
                            "ERROR",
                            t.task_id,
                            f"{t.task_id}: banned token {tok!r} in agent-visible {where}",
                        )
                    )
    return out


def check_fnv_contract(ctx: PreflightContext) -> list[Finding]:
    """Require participant-visible documentation of the FNV basis used by graders.

    The documented basis allows submissions to produce matching digest values.
    """
    out = []
    by_id = {r["task_id"]: r for r in ctx.stats.rows}
    for t in _per_task(ctx):
        bundle = ctx.tree.bundle(t.task_id)
        folds = False
        for f in grader_files(bundle):
            low = _read(f).lower()
            if any(tok in low for tok in _FNV_TOKENS):
                folds = True
                break
        if not folds:
            continue
        contract = by_id.get(t.task_id, {}).get("prompt", "")
        if (bundle / "README.md").exists():
            contract += _read(bundle / "README.md")
        contract += "".join(
            _read(f)
            for f in bundle.iterdir()
            if f.is_file() and f.suffix in (".h", ".hpp", ".cuh", ".py")
        )
        if not any(tok in contract.lower() for tok in _FNV_TOKENS):
            out.append(
                Finding(
                    "fnv_contract",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: grader folds the project FNV-1a-64 basis but the "
                    f"agent-visible contract never states it",
                )
            )
    return out


def check_compat_golden(ctx: PreflightContext) -> list[Finding]:
    """Verify hashes for participant-visible strings with compatibility guarantees."""
    return [
        Finding(
            "compat_golden",
            "ERROR",
            None,
            f"frozen agent-visible string {name} drifted from its golden sha256 — "
            f"the graded/grepped surface changed",
        )
        for name in markers.check_golden()
    ]


def check_score_images(ctx: PreflightContext) -> list[Finding]:
    """Report image availability and whether registry lookup remains possible."""
    images: dict[str, str] = {}
    for task in ctx.tasks:
        kind = runtime_kind_of(task)
        images.setdefault(kind, resolve_score_image(ctx.config, kind))
    if ctx.config.score_runtime_type == "modal":
        findings = []
        if error := modal_prerequisite_error():
            findings.append(Finding("score_images", "ERROR", None, error))
        if error := modal_gpu_arch_error(ctx.config.arch, ctx.config.score_modal_gpu):
            findings.append(Finding("score_images", "ERROR", None, error))
        if findings:
            return findings
        for kind, image in sorted(images.items()):
            if not modal_image_is_registry_addressable(image):
                findings.append(
                    Finding(
                        "score_images",
                        "ERROR",
                        None,
                        f"required {kind} Modal scoring image {image} is not "
                        "remote-registry addressable",
                    )
                )
                continue
        return findings
    docker = shutil.which("docker")
    if docker is None:
        return [
            Finding(
                "score_images",
                "ERROR",
                None,
                "docker not found — release scoring cannot run",
            )
        ]
    try:
        daemon = subprocess.run([docker, "info"], capture_output=True, timeout=10)
    except Exception as error:  # noqa: BLE001 - convert probe failures into findings
        return [
            Finding(
                "score_images",
                "ERROR",
                None,
                f"docker daemon probe failed ({type(error).__name__})",
            )
        ]
    if daemon.returncode != 0:
        return [
            Finding(
                "score_images",
                "ERROR",
                None,
                "docker daemon is unavailable — release scoring cannot run",
            )
        ]
    findings = []
    for kind, image in sorted(images.items()):
        try:
            result = subprocess.run(
                [docker, "image", "inspect", image], capture_output=True, timeout=10
            )
        except Exception as error:  # noqa: BLE001 - convert probe failures into findings
            findings.append(
                Finding(
                    "score_images",
                    "ERROR",
                    None,
                    f"docker image probe failed for {image} ({type(error).__name__})",
                )
            )
            continue
        if result.returncode == 0:
            continue
        local_only = not score_image_is_pullable(image)
        findings.append(
            Finding(
                "score_images",
                "ERROR" if local_only else "WARN",
                None,
                f"required {kind} scoring image {image} is not present locally"
                + (
                    "; build it before loading the release taskset"
                    if local_only
                    else "; Docker may pull it at run time"
                ),
            )
        )
    return findings


# Advisory checks


def check_pool_tree_coherence(ctx: PreflightContext) -> list[Finding]:
    """Verify bundle/reference membership and reject extra sanity task directories."""
    pool_ids = {r["task_id"] for r in ctx.stats.rows}
    out = []
    for kind, present in (
        ("bundle", set(ctx.tree.bundle_ids())),
        ("reference", set(ctx.tree.reference_ids())),
    ):
        if missing := sorted(pool_ids - present):
            out.append(
                Finding(
                    "pool_tree_coherence",
                    "ERROR",
                    None,
                    f"{kind} tree is missing release task(s): {missing}",
                )
            )
        if extras := sorted(present - pool_ids):
            out.append(
                Finding(
                    "pool_tree_coherence",
                    "ERROR",
                    None,
                    f"{kind} tree contains non-release task(s): {extras}",
                )
            )
    sanity_extras = sorted(set(ctx.tree.sanity_ids()) - pool_ids)
    if sanity_extras:
        out.append(
            Finding(
                "pool_tree_coherence",
                "ERROR",
                None,
                f"sanity tree contains non-release task(s): {sanity_extras}",
            )
        )
    return out


def check_pycache_hygiene(ctx: PreflightContext) -> list[Finding]:
    junk = [p for p in ctx.tree.root.rglob("__pycache__")] + [
        p for p in ctx.tree.root.rglob("*.pyc")
    ]
    if not junk:
        return []
    return [
        Finding(
            "pycache_hygiene",
            "WARN",
            None,
            f"data tree contains {len(junk)} __pycache__/*.pyc artifact(s) — "
            f"remove them before cutting the release artifact",
        )
    ]


def _strip_comments(text: str, is_py: bool) -> str:
    if is_py:
        text = re.sub(r"(?m)#[^\n]*", "", text)
    else:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def check_common_h_drift(ctx: PreflightContext) -> list[Finding]:
    """Compare shared bundle and sanity files without comments or whitespace.

    Comment-only differences are informational. Code differences are errors because
    the two environments would expose different interfaces.
    """
    out = []
    doc_only = 0
    for t in _per_task(ctx):
        sd = ctx.tree.sanity_dir(t.task_id)
        if not sd.is_dir():
            continue
        for f in ctx.tree.bundle(t.task_id).iterdir():
            if f.name.startswith(("test_", "bench_")) or f.suffix not in (
                ".h",
                ".hpp",
                ".cuh",
                ".py",
            ):
                continue
            sf = sd / f.name
            if not (f.is_file() and sf.exists()) or sf.read_bytes() == f.read_bytes():
                continue
            is_py = f.suffix == ".py"
            if _strip_comments(_read(f), is_py) == _strip_comments(_read(sf), is_py):
                doc_only += 1
            else:
                out.append(
                    Finding(
                        "common_h_drift",
                        "ERROR",
                        t.task_id,
                        f"{t.task_id}: {f.name} CODE drifted between bundle and sanity — "
                        f"agent compiles against a different contract than the scorer",
                    )
                )
    if doc_only:
        out.append(
            Finding(
                "common_h_drift",
                "INFO",
                None,
                f"{doc_only} bundle-vs-sanity shared file(s) drift in comments only "
                f"(known doc-only carve-out)",
            )
        )
    return out


def check_gate_verdict(ctx: PreflightContext) -> list[Finding]:
    """Validate optional feasibility verdicts for performance-gated tasks.

    A verdict that the gate is ineffective is an error. Missing per-task verdicts are
    warnings, while an absent sidecar is allowed.
    """
    p = ctx.tree.root / "gate_verdicts.json"
    if not p.exists():
        return []
    verdicts = _load_sidecar(p, "gate_verdict", want=dict)
    out = []
    for t in _per_task(ctx):
        if not perf_enabled(ctx.config, t):
            continue
        v = verdicts.get(t.task_id)
        if v is None:
            out.append(
                Finding(
                    "gate_verdict",
                    "WARN",
                    t.task_id,
                    f"{t.task_id}: perf-gated but no feasibility verdict recorded",
                )
            )
        elif isinstance(v, dict) and str(v.get("verdict", "")).replace("-", "_") in (
            "inherently_serial",
            "bandwidth_bound",
        ):
            out.append(
                Finding(
                    "gate_verdict",
                    "ERROR",
                    t.task_id,
                    f"{t.task_id}: gate verdict {v.get('verdict')!r} says the perf gate "
                    f"cannot bite, but the pool row is perf-gated",
                )
            )
    return out


def check_label_freshness(ctx: PreflightContext) -> list[Finding]:
    """Verify that optional difficulty labels match the current pool hash."""
    p = ctx.tree.root / "calibration_labels.json"
    if not p.exists():
        return []
    rec = _load_sidecar(p, "label_freshness", want=dict)
    if rec.get("pool_hash") == ctx.pool_hash:
        return []
    sev = "ERROR" if ctx.config.preflight_strict_labels else "WARN"
    return [
        Finding(
            "label_freshness",
            sev,
            None,
            f"calibration_labels.json was recorded for pool_hash "
            f"{str(rec.get('pool_hash'))[:16]}… but the current pool is "
            f"{ctx.pool_hash[:16]}… — difficulty labels expired",
        )
    ]


def check_prompt_paths(ctx: PreflightContext) -> list[Finding]:
    """Warn when prompts name source files absent from the provisioned workspace."""
    out = []
    by_id = {r["task_id"]: r for r in ctx.stats.rows}
    for t in _per_task(ctx):
        bundle = ctx.tree.bundle(t.task_id)
        present = {
            t.student_file,
            "README.md",
            "verify_student",
            "paired_selfcheck.sh",
            "Makefile",
            "GRADER_HIDDEN",
        }
        present.update(f.name for f in grader_files(bundle))
        sd = ctx.tree.sanity_dir(t.task_id)
        if sd.is_dir():
            present.update(f.name for f in grader_files(sd))
        prompt = by_id.get(t.task_id, {}).get("prompt", "")
        mentioned = set(
            re.findall(r"`\.?/?([\w][\w.\-]*\.(?:cu|py|h|hpp|cuh|sh))`", prompt)
        )
        ghosts = {m for m in mentioned if m not in present}
        if ghosts:
            out.append(
                Finding(
                    "prompt_paths",
                    "WARN",
                    t.task_id,
                    f"{t.task_id}: prompt mentions file(s) not provisioned in the "
                    f"workspace: {sorted(ghosts)}",
                )
            )
    return out


def check_correctness_inputs(ctx: PreflightContext) -> list[Finding]:
    findings = []
    for task in _per_task(ctx):
        if task.task_id not in grader_inputs.RANDOMIZED_TASKS:
            continue
        files = grader_files(ctx.tree.bundle(task.task_id))
        try:
            grader_inputs.validate_filename(task.task_id, [f.name for f in files])
            for f in files:
                if f.name.startswith("test_"):
                    grader_inputs.render(
                        task.task_id, f.read_bytes(), 0xD7B94CA163E5082F
                    )
        except PoolIntegrityError as error:
            findings.append(
                Finding("correctness_inputs", "ERROR", task.task_id, str(error))
            )
    return findings


ALL_CHECKS: tuple[tuple[str, Callable], ...] = (
    ("compat_golden", check_compat_golden),
    ("starter_exists", check_starter_exists),
    ("makefile_exists", check_makefile_exists),
    ("sanity_overclaim", check_sanity_overclaim),
    ("stub_leak", check_stub_leak),
    ("canary_integrity", check_canary_integrity),
    ("paired_bench_format", check_paired_bench_format),
    ("paired_ref_exists", check_paired_ref_exists),
    ("makefile_targets", check_makefile_targets),
    ("bench_target_consistency", check_bench_target_consistency),
    ("student_file_consistency", check_student_file_consistency),
    ("sanity_leak", check_sanity_leak),
    ("sanity_ref_leak", check_sanity_ref_leak),
    ("sanity_stale", check_sanity_stale),
    ("correctness_inputs", check_correctness_inputs),
    ("effective_pool", check_effective_pool),
    ("self_check_coverage", check_self_check_coverage),
    ("perf_census", check_perf_census),
    ("answer_key_lint", check_answer_key_lint),
    ("fnv_contract", check_fnv_contract),
    ("score_images", check_score_images),
    ("pool_tree_coherence", check_pool_tree_coherence),
    ("pycache_hygiene", check_pycache_hygiene),
    ("common_h_drift", check_common_h_drift),
    ("gate_verdict", check_gate_verdict),
    ("label_freshness", check_label_freshness),
    ("prompt_paths", check_prompt_paths),
)


@dataclasses.dataclass
class PreflightReport:
    findings: list[Finding]
    lever: LeverVector

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    def header_lines(self) -> list[str]:
        lines = list(self.lever.header_lines())
        for f in self.findings:
            if f.severity != "ERROR":
                lines.append(f"PMPP preflight {f.severity} [{f.check}]: {f.message}")
        if self.errors:
            lines.append(f"PMPP preflight: {len(self.errors)} ERROR(s)")
        return lines

    def raise_if_failed(self) -> None:
        problems = [f.message for f in self.errors]
        if problems:
            raise PoolIntegrityError(
                f"{markers.PREFLIGHT_FAIL_PREFIX} ({len(problems)} problem(s); "
                f"preflight=false only for debugging):\n  " + "\n  ".join(problems)
            )


def run_preflight(
    ctx: PreflightContext, skip_checks: frozenset[str] = frozenset()
) -> PreflightReport:
    """Run all checks and apply configured severity waivers."""
    missing = frozenset(f.task_id for f in check_starter_exists(ctx) if f.task_id)
    ctx = dataclasses.replace(ctx, missing_starters=missing)
    findings: list[Finding] = []
    waived = set(ctx.config.preflight_waive or [])
    for name, fn in ALL_CHECKS:
        if name in skip_checks:
            continue
        try:
            results = fn(ctx)
        except _SidecarError as e:
            # Convert sidecar parsing failures to findings so waiver handling still applies.
            results = [Finding(name, "ERROR", None, f"sidecar load failed: {e}")]
        for f in results:
            if f.severity == "ERROR" and name in waived:
                f = dataclasses.replace(
                    f, severity="WARN", message=f"[waived] {f.message}"
                )
            findings.append(f)
    lever = LeverVector.compute(
        ctx.config, ctx.tasks, ctx.tree, ctx.pool_hash, perf_enabled
    )
    return PreflightReport(findings=findings, lever=lever)
