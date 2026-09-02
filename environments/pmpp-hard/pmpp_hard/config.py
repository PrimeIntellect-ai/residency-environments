"""Task metadata, environment settings, and comparable-run configuration."""

import dataclasses
from typing import Literal

import verifiers.v1 as vf

GATE_PERF = {"perf", "insight"}  # gate types that require performance scoring
RELEASE_MANIFEST = "pool_pipeline_pmpp_hard69.jsonl"
RELEASE_SANITY_MANIFEST = "sanity_manifest_pmpp_hard69.json"

# Runtime-specific scoring images. Unlisted runtimes use ``score_image``.
DEFAULT_SCORE_IMAGES: dict[str, str] = {
    "triton": "pmpp-triton:cu128",
    "cutlass": "pmpp-cutlass:12.8.1",
}


class PMPPHardTaskData(vf.TaskData):
    task_id: str
    student_file: str
    gate_type: Literal["perf", "insight", "correctness", "precision", "bit_exactness"]
    test_target: str = "test_student"
    bench_target: str = "bench_student"
    perf_gated: bool = True
    sanity_ok: bool = False  # whether a validated held-out grader is available
    category: str | None = None
    family: str | None = None
    runtime_kind: Literal["cuda", "triton", "cutlass"]
    # A per-task target overrides the environment-wide paired-ratio target.
    perf_ratio_target: float | None = None


class PMPPHardConfig(vf.TasksetConfig, vf.TaskConfig):
    # Reject unknown settings so misspelled or retired options cannot be ignored.
    model_config = {"extra": "forbid"}
    manifest: Literal["pool_pipeline_pmpp_hard69.jsonl"] = RELEASE_MANIFEST
    include_ids: list[str] | None = None
    require_perf: bool = True  # disable all performance gates when false
    arch: str = "sm_120"
    # Use a separate grader for agent feedback and score in a clean runtime.
    held_out: bool = (
        True  # expose the held-out sanity grader instead of the scoring grader
    )
    sanity_manifest: Literal["sanity_manifest_pmpp_hard69.json"] = (
        RELEASE_SANITY_MANIFEST
    )
    # Optional per-task agent images override the harness runtime image. Modal
    # deployments use registry references here; existing harness configs remain
    # authoritative when these fields are unset.
    agent_image: str | None = None
    agent_images: dict[str, str] = {}
    score_runtime_type: Literal["docker", "modal"] = "docker"
    score_gpu: str = "1"  # GPU count or device request for the scoring container
    # RTX-PRO-6000 is Modal's compute-capability 12.0 GPU and matches ``sm_120``.
    score_modal_gpu: str = "RTX-PRO-6000"
    score_modal_region: str | None = None
    score_modal_network_access: bool = False
    score_modal_creates_per_sec: float | None = 10.0
    score_image: str = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
    score_cpu: float = 8.0
    score_memory: float = 32.0
    perf_ratio_target: float = (
        1.25  # maximum accepted median student-to-reference timing ratio
    )
    perf_ratio_runs: int = 5  # number of interleaved timing pairs
    gpu_lock: bool = True  # serialize scoring runs that share a GPU
    # Remind the agent to write its best available solution to the submission file.
    always_submit: bool = True
    single_shot: bool = (
        False  # use one attempt without iterative grader feedback when true
    )
    preflight: bool = True  # validate package integrity before loading tasks
    require_self_check: bool = (
        False  # require at least one self-checkable task when applicable
    )
    expected_perf_gated: int | None = (
        None  # validate the effective number of performance-gated tasks
    )
    preflight_strict_labels: bool = (
        False  # treat stale calibration labels as preflight errors
    )
    preflight_waive: list[
        str
    ] = []  # checks whose preflight errors should be reported as warnings
    # Stop after this many consecutive scoring infrastructure failures; zero disables it.
    score_breaker: int = 0
    # Require observable GPU activity for tasks without a performance gate.
    residency_check: bool = True
    residency_min_launches: int = (
        1  # minimum kernel launches observed during the probed run
    )
    residency_gpu_frac: float = (
        0.02  # minimum GPU-time ratio on the same input; zero disables the floor
    )
    # Reject captured source that accesses reference implementations or alters probes.
    source_policy: bool = True
    # Require a Triton JIT kernel to be launched and observed at runtime.
    triton_premise: bool = True
    # Apply supplementary source and timing checks during clean scoring.
    kernelguard_check: bool = True
    kernelguard_extreme_speedup: float = (
        10.0  # record larger student-to-reference speedups for review
    )
    # Treat lower student-to-reference timing ratios as performance anomalies.
    kernelguard_ratio_floor: float = 0.05
    # Unlisted runtimes fall back to ``score_image``; an empty map uses it globally.
    score_images: dict[str, str] = DEFAULT_SCORE_IMAGES


@dataclasses.dataclass(frozen=True)
class LeverVector:
    """Settings and pool identity required when comparing scores."""

    held_out: bool
    clean_score: bool
    single_shot: bool
    require_perf: bool
    perf_mode: str
    perf_ratio_target: float
    hidden_shape: bool
    only_supported: bool
    arch: str
    score_runtime_type: str
    score_gpu: str
    manifest: str
    pool_hash: str
    n_tasks: int
    n_perf_gated: int
    n_self_checkable: int
    n_hidden_shape: int
    # Enforcement settings also affect score comparability.
    residency_check: bool = True
    source_policy: bool = True
    triton_premise: bool = True
    # Supplementary policy checks are part of the scoring contract.
    kernelguard_check: bool = True

    @classmethod
    def compute(
        cls, config: PMPPHardConfig, tasks: list, tree, pool_hash: str, perf_enabled
    ) -> "LeverVector":
        return cls(
            # Keep compatibility fields stable for existing result readers.
            held_out=config.held_out,
            clean_score=True,
            single_shot=config.single_shot,
            require_perf=config.require_perf,
            perf_mode="paired_ratio",
            perf_ratio_target=config.perf_ratio_target,
            hidden_shape=False,
            only_supported=False,
            arch=config.arch,
            score_runtime_type=config.score_runtime_type,
            score_gpu=(
                config.score_modal_gpu
                if config.score_runtime_type == "modal"
                else config.score_gpu
            ),
            manifest=config.manifest,
            pool_hash=pool_hash,
            n_tasks=len(tasks),
            n_perf_gated=sum(1 for t in tasks if perf_enabled(config, t)),
            n_self_checkable=sum(
                1 for t in tasks if t.sanity_ok and tree.sanity_dir(t.task_id).is_dir()
            ),
            n_hidden_shape=0,
            residency_check=config.residency_check,
            source_policy=config.source_policy,
            triton_premise=config.triton_premise,
            kernelguard_check=config.kernelguard_check,
        )

    def header_lines(self) -> list[str]:
        return [
            f"PMPP pool: {self.manifest}  tasks={self.n_tasks}  pool_hash={self.pool_hash[:16]}",
            f"PMPP levers: held_out={self.held_out} clean_score={self.clean_score} "
            f"single_shot={self.single_shot} require_perf={self.require_perf} "
            f"perf_mode={self.perf_mode} target={self.perf_ratio_target} "
            f"hidden_shape={self.hidden_shape} only_supported={self.only_supported} arch={self.arch} "
            f"score_runtime={self.score_runtime_type} score_gpu={self.score_gpu} "
            f"residency={self.residency_check} source_policy={self.source_policy} "
            f"triton_premise={self.triton_premise} kernelguard={self.kernelguard_check}",
            f"PMPP coverage: {self.n_self_checkable}/{self.n_tasks} tasks self-checkable, "
            f"{self.n_perf_gated} perf-gated, {self.n_hidden_shape} hidden-shape",
        ]

    def as_json(self) -> dict:
        return dataclasses.asdict(self)
