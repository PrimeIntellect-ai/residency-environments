"""PMPP Hard — hard GPU-kernel tasks on the Verifiers v1 GPU-sandbox runtime.

Each task ships a behavioral contract (the prompt) and an opaque grader (a Makefile +
test_* + bench_*, never the reference). The model produces a kernel source; the taskset
compiles it, runs the correctness test, and — for perf-gated tasks — runs the benchmark
as a paired student/reference ratio in a clean scoring sandbox the agent never
touched. Run it agentically (a coding harness edits the file and self-checks via ./verify_student)
or single-turn (one fenced code block, extracted in finalize).

This file is deliberately thin lifecycle glue: no agent-visible string literals (those
live golden-hashed in markers.py) and no filesystem-walking logic (paths.py/pool.py).
Every release invariant is a named preflight check in preflight.py.

Lifecycle and scoring live on PMPPHardTask, while PMPPHardTaskset owns release-pool
loading and validation. The taskset injects the shared DataTree, LeverVector, and
per-task bundle hash onto each task instance after construction.
"""

import verifiers.v1 as vf

from pmpp_hard import capture, pool, provision, scoring
from pmpp_hard.config import (  # noqa: F401 (GATE_PERF re-export)
    GATE_PERF,
    PMPPHardConfig,
    PMPPHardTaskData,
)
from pmpp_hard.paths import DataTree, Workspace
from pmpp_hard.preflight import PreflightContext, run_preflight


class PMPPHardTask(vf.Task[PMPPHardTaskData, vf.State, PMPPHardConfig]):
    NEEDS_CONTAINER = True

    def __init__(
        self, data: PMPPHardTaskData, config: PMPPHardConfig | None = None
    ) -> None:
        super().__init__(data, config)
        # injected by PMPPHardTaskset.load() after construction (shared across tasks):
        self._tree = DataTree()
        self._lever = None
        self._bundle_hash: str | None = None
        self._setup_outcomes: dict[
            int, provision.SetupOutcome
        ] = {}  # keyed by id(runtime):
        # unique per rollout, so a group's concurrent rollouts (which share this task
        # instance) never cross-attribute

    @property
    def key(self) -> str:
        return self.data.task_id

    # ── per-rollout prep: build the sandbox workspace (visible stub + grader) ──
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await scoring.ensure_score_image_ready(self.config, self.data)
        outcome = await provision.setup_workspace(
            self.config, self.data, Workspace(runtime), self._tree
        )
        self._setup_outcomes[id(runtime)] = (
            outcome  # host-side only; never in the sandbox
        )

    # ── after the harness: ensure a student file is on disk + snapshot provenance ──
    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        info = trace.info.setdefault("pmpp", {})
        if self._lever is not None:
            info["lever_vector"] = self._lever.as_json()
        info["bundle_sha256"] = self._bundle_hash
        outcome = self._setup_outcomes.pop(id(runtime), None)
        if outcome is not None:
            info["setup_mode"] = outcome.mode
            if outcome.degraded_reason:
                info["setup_degraded"] = outcome.degraded_reason
        await capture.capture_submission(self.data, trace, runtime, self._tree)

    # ── grade: compile + run + (perf) benchmark against the REAL grader; parse markers ──
    @vf.metric
    async def verify(
        self, task: PMPPHardTaskData, trace: vf.Trace, runtime: vf.Runtime
    ) -> dict:
        return await scoring.score_clean(self.config, task, trace, runtime, self._tree)

    @vf.reward(weight=1.0)
    async def success(self, trace: vf.Trace) -> float:
        m = trace.metrics
        return (
            1.0
            if (m.get("build_pass") and m.get("run_pass") and m.get("performance_pass"))
            else 0.0
        )


class PMPPHardTaskset(vf.Taskset[PMPPHardTask, PMPPHardConfig]):
    def __init__(self, config: PMPPHardConfig) -> None:
        super().__init__(config)
        self._tree = DataTree()
        self._lever = None
        self._bundle_hashes: dict[str, str] = {}

    def load(self) -> list[PMPPHardTask]:
        rows, stats = pool.build_tasks(self.config, self._tree)
        self._bundle_hashes = {
            r.task_id: pool.bundle_sha256(self._tree, r.task_id) for r in rows
        }
        pool_hash = pool.pool_content_hash(
            self._tree, rows, stats.rows, self._bundle_hashes
        )
        report = run_preflight(
            PreflightContext(
                tree=self._tree,
                config=self.config,
                tasks=rows,
                stats=stats,
                pool_hash=pool_hash,
            )
        )
        print(
            "\n".join(report.header_lines())
        )  # ALWAYS — even with preflight=false (A1/B6)
        if self.config.preflight:
            report.raise_if_failed()
        elif report.errors:
            print(
                f"PMPP preflight: {len(report.errors)} ERROR(s) IGNORED (preflight=false)"
            )
        self._lever = report.lever
        tasks: list[PMPPHardTask] = []
        for row in rows:
            task = PMPPHardTask(row, self.config)
            task._tree = self._tree
            task._lever = self._lever
            task._bundle_hash = self._bundle_hashes.get(row.task_id)
            tasks.append(task)
        return tasks
