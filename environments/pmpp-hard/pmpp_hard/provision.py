"""Assemble agent and scoring workspaces.

`setup_workspace` writes the task stub and README before selecting the configured
grading mode. If sanity data is unavailable, setup uses opaque grading and records
the reason on the host without exposing it in the agent workspace.
"""

import dataclasses
import pathlib
from typing import Literal

from pmpp_hard import grader_inputs, markers
from pmpp_hard.config import PMPPHardConfig, PMPPHardTaskData
from pmpp_hard.errors import ScoreInfraError
from pmpp_hard.paths import DataTree, Workspace

SetupMode = Literal["grader_sanity", "grader_real", "opaque", "single_shot"]

# Exclude answer-bearing grader assets even when their suffix is otherwise allowed.
_DENY_PREFIXES = ("reference_solution", "reference_oracle", "reference_triton", "naive")


def provisionable(f: pathlib.Path) -> bool:
    """Return whether an entrypoint, shared header, or Python helper may be copied.

    Answer-bearing and validation files are excluded.
    """
    if not f.is_file() or f.suffix == ".pyc":
        return False
    if f.name.startswith(_DENY_PREFIXES) or "VALIDATION" in f.name:
        return False
    return (
        f.name.startswith("test_")
        or f.name.startswith("bench_")
        or f.suffix in (".h", ".hpp", ".cuh", ".py")
    )


def grader_files(src: pathlib.Path) -> list[pathlib.Path]:
    """Return the grader files that would be visible to the agent."""
    return [f for f in src.iterdir() if provisionable(f)]


def render_verify_script(task: PMPPHardTaskData) -> str:
    return markers.VERIFY_SCRIPT.format(
        student_file=task.student_file,
        test_target=task.test_target,
        bench_target=task.bench_target,
    )


@dataclasses.dataclass(frozen=True)
class SetupOutcome:
    """Host-side record of the provisioned mode and any fallback reason."""

    mode: SetupMode
    degraded_reason: str | None = None


def perf_enabled(config: PMPPHardConfig, task: PMPPHardTaskData) -> bool:
    """Return whether the task uses the paired-ratio performance gate."""
    return task.perf_gated


async def provision_grader(
    config: PMPPHardConfig,
    task: PMPPHardTaskData,
    ws: Workspace,
    src: pathlib.Path,
    tree: DataTree,
    with_ref_binary: bool = False,
    *,
    correctness_parameter: int | None = None,
) -> str | None:
    """Populate a grader from sanity or authoritative bundle files.

    The caller owns the student file. When requested, a source-stripped reference
    benchmark is built for agent-side CUDA performance checks. Python references stay
    private because their source cannot be embedded in an agent-visible executable. A
    build failure returns a host-side reason that is not written into the workspace.
    """
    bundle = tree.bundle(task.task_id)
    if correctness_parameter is not None:
        if src != bundle:
            raise ValueError(
                "randomized correctness inputs belong only in the scoring grader"
            )
        grader_inputs.validate_filename(task.task_id, [f.name for f in src.iterdir()])
    await ws.write(".grader/Makefile", (bundle / "Makefile").read_bytes())
    for f in src.iterdir():
        if not provisionable(f):
            continue
        content = f.read_bytes()
        if correctness_parameter is not None and f.name.startswith("test_"):
            content = grader_inputs.render(task.task_id, content, correctness_parameter)
        await ws.write(f".grader/{f.name}", content)
    perf = "1" if perf_enabled(config, task) else "0"
    await ws.write("verify_student", render_verify_script(task).encode())
    await ws.run(["chmod", "+x", "/app/verify_student"])
    await ws.write(".grader/.perf", perf.encode())
    await ws.write(".grader/.nvccflags", f"-O3 -std=c++17 -arch={config.arch}".encode())
    await ws.write(
        ".grader/.perf_target",
        f"{task.perf_ratio_target or config.perf_ratio_target}".encode(),
    )
    await ws.write(".grader/paired_selfcheck.sh", markers.PAIRED_SELFCHECK.encode())
    if with_ref_binary and perf == "1" and task.student_file.endswith(".cu"):
        ref = tree.find_reference(task.task_id, cuda_only=True)
        if ref is not None:
            await ws.write(".grader/reference_solution.cu", ref.read_bytes())
            await ws.run(
                [
                    "bash",
                    "-lc",
                    "cd /app/.grader && make bench_reference PY=python3 "
                    'NVCCFLAGS="$(cat .nvccflags)" >/tmp/refbuild.log 2>&1; '
                    "rm -f reference_solution.cu",
                ]
            )
            probe = await ws.run(
                ["bash", "-lc", "test -x /app/.grader/bench_reference"]
            )
            if getattr(probe, "exit_code", 0) != 0:
                return (
                    "agent-side bench_reference failed to build — perf self-check "
                    "is unavailable"
                )
    return None


async def setup_workspace(
    config: PMPPHardConfig, task: PMPPHardTaskData, ws: Workspace, tree: DataTree
) -> SetupOutcome:
    """Provision the configured workspace and report the effective grading mode."""
    bundle = tree.bundle(task.task_id)
    await ws.write(task.student_file, (bundle / task.student_file).read_bytes())
    readme = bundle / "README.md"
    if readme.exists():
        await ws.write("README.md", readme.read_bytes())
    outcome = SetupOutcome(mode="grader_real")
    if config.single_shot:
        await ws.write("GRADER_HIDDEN", markers.GRADER_HIDDEN_SINGLE_SHOT)
        outcome = SetupOutcome(mode="single_shot")
    elif config.held_out:
        if task.sanity_ok and tree.sanity_dir(task.task_id).is_dir():
            degrade = await provision_grader(
                config,
                task,
                ws,
                tree.sanity_dir(task.task_id),
                tree,
                with_ref_binary=True,
            )
            outcome = SetupOutcome(mode="grader_sanity", degraded_reason=degrade)
        else:
            await ws.write("GRADER_HIDDEN", markers.GRADER_HIDDEN_OPAQUE)
            reason = (
                "sanity manifest claims valid but data/sanity dir missing — degraded to opaque"
                if task.sanity_ok
                else None
            )
            outcome = SetupOutcome(mode="opaque", degraded_reason=reason)
    else:
        await provision_grader(config, task, ws, bundle, tree)
    # Python dependencies are needed only when the agent receives a runnable grader.
    # Other modes do not need to probe the sandbox.
    if task.student_file.endswith(".py"):
        if outcome.mode in ("grader_sanity", "grader_real"):
            probe = await ws.run(
                ["bash", "-lc", "python3 -c 'import torch, triton' 2>&1"]
            )
            if getattr(probe, "exit_code", 0) != 0:
                tail = (
                    getattr(probe, "stdout", "") + getattr(probe, "stderr", "")
                ).strip()[-300:]
                raise ScoreInfraError(
                    f"agent sandbox image cannot import torch/triton for {task.task_id}; "
                    f"rebuild the Triton image with scripts/pmpp-hard/build-images.sh: {tail}"
                )
        else:
            outcome = dataclasses.replace(
                outcome,
                degraded_reason=(
                    outcome.degraded_reason
                    or "agent-side Python dependency probe skipped "
                    f"({outcome.mode}: no agent grader needs torch/triton)"
                ),
            )
    return outcome
