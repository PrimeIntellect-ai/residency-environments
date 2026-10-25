"""The agent's scorer: train on a corpus snapshot and measure loss on the public part of the held-out set."""

import asyncio
import contextlib
import json
import logging
import math
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import verifiers.v1 as vf
from prime_sandboxes import AsyncSandboxClient

from .config import ScorerToolsetConfig, ScoringConfig
from .scoring import CorpusError, Job, JobResult, ScorerVM, model_spec
from .state import PretrainState
from .workspace import DockerBox, PrimeBox, open_box

logger = logging.getLogger(__name__)
CURVE_POINTS = 20
MAX_WAIT_SECONDS = 3600.0


def train_loss_curve(steps: list[dict]) -> list[list[float]]:
    """`[tokens trained, mean training loss since the previous point]` at about CURVE_POINTS points."""
    size = max(1, len(steps) // CURVE_POINTS)
    return [
        [steps[end - 1]["tokens"], statistics.fmean(step["train_loss"] for step in steps[start:end])]
        for start in range(0, len(steps), size)
        if (end := min(start + size, len(steps)))
    ]


def public_metrics(metrics: list[dict]) -> list[dict]:
    """The trainer's records with each periodic evaluation reduced to its mean validation loss."""
    return [
        dict(step=record["step"], tokens=record["tokens"], val_loss=statistics.fmean(record["quick_eval"].values()))
        if "quick_eval" in record
        else record
        for record in metrics
    ]


def summarize(config: ScoringConfig, result: JobResult, metrics: list[dict]) -> dict:
    steps = [record for record in metrics if "train_loss" in record]
    evals = [record for record in public_metrics(metrics) if "val_loss" in record]
    train = result.train
    tail = steps[-max(1, len(steps) // 20) :]
    # The first step includes compilation, so throughput is measured over the steps after it.
    if len(steps) > 1:
        steady_tokens = steps[-1]["tokens"] - steps[0]["tokens"]
        steady_seconds = sum(step["step_seconds"] for step in steps[1:])
    else:
        steady_tokens, steady_seconds = train["train_tokens"], train["train_seconds"]
    return dict(
        val_loss=result.loss,
        estimated_reward=config.performance(result.loss, result.vocab_size),
        train_tokens=train["train_tokens"],
        steps=train["steps"],
        diverged=train["diverged"],
        final_train_loss=statistics.fmean(step["train_loss"] for step in tail) if tail else None,
        train_loss_curve=train_loss_curve(steps),
        # Periodic evaluations use a sample of the validation text; the last point is the full evaluation.
        val_loss_curve=[[record["tokens"], record["val_loss"]] for record in evals]
        + ([[train["train_tokens"], result.loss]] if math.isfinite(result.loss) else []),
        mean_grad_norm=statistics.fmean(step["grad_norm"] for step in steps) if steps else None,
        tokens_per_second=steady_tokens / steady_seconds if steady_seconds else None,
        first_step_seconds=steps[0]["step_seconds"] if steps else None,
        train_seconds=train["train_seconds"],
        # Counts that depend on how much held-out text was removed would reveal what the set contains.
        short_documents_skipped=result.prepare["documents_short"],
    )


@dataclass
class Probe:
    id: str
    job: Job
    workspace: DockerBox | PrimeBox
    submitted: float = field(default_factory=time.time)
    task: asyncio.Task | None = None
    summary: dict | None = None


class ScorerToolset(vf.Toolset[ScorerToolsetConfig, PretrainState]):
    TOOL_PREFIX = "scorer"

    def __init__(self, config: ScorerToolsetConfig) -> None:
        super().__init__(config)
        self.scoring = config.scoring
        self.host_dir = config.host_dir.expanduser()
        self.batch_tokens = model_spec(self.scoring)["train"]["batch_tokens"]
        self.probes: dict[str, Probe] = {}
        self._client: AsyncSandboxClient | None = None
        self._vm: ScorerVM | None = None
        self._release: asyncio.Task | None = None
        # Parallel `score` calls would otherwise both start a job.
        self._submitting = asyncio.Lock()

    async def setup(self) -> None:
        self._exit_stack.push_async_callback(self._teardown)

    async def _teardown(self) -> None:
        tasks = [probe.task for probe in self.probes.values() if probe.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._release is not None:
            self._release.cancel()
        await self._drop_vm()
        if self._client is not None:
            await self._client.aclose()

    def _steps(self, train_tokens: int) -> int:
        """Optimizer steps a run of `train_tokens` takes; it trains on whole batches."""
        return max(1, train_tokens // self.batch_tokens)

    def _workspace(self) -> DockerBox | PrimeBox:
        if self.state.workspace_kind == "prime" and self._client is None:
            self._client = AsyncSandboxClient()
        return open_box(self.state.workspace_kind, self.state.workspace, self._client)

    async def _acquire_vm(self) -> ScorerVM | None:
        """The warm scorer VM on the Prime backend, started if there is none."""
        if self.scoring.scorer.backend != "prime":
            return None
        if self._release is not None:
            self._release.cancel()
            self._release = None
        if self._vm is None:
            vm = ScorerVM(self.scoring.scorer.prime)
            try:
                await vm.start()
            except BaseException:
                await vm.stop()
                raise
            self._vm = vm
        return self._vm

    async def _drop_vm(self) -> None:
        vm, self._vm = self._vm, None
        if vm is not None:
            await vm.stop()

    async def _release_later(self) -> None:
        await asyncio.sleep(self.scoring.scorer.idle_release_seconds)
        self._release = None
        await self._drop_vm()

    async def _run(self, probe: Probe) -> None:
        directory = probe.job.directory
        try:
            probe.job.stage = "starting the scorer"
            result = await probe.job.run(await self._acquire_vm())
            metrics = [json.loads(line) for line in (directory / "out" / "metrics.jsonl").read_text().splitlines()]
            probe.summary = dict(job_id=probe.id, status="done", **summarize(self.scoring, result, metrics))
        except CorpusError as exc:
            probe.summary = dict(job_id=probe.id, status="failed", error=f"corpus: {exc}")
        except Exception:
            # A failed probe is reported to the agent; the rollout continues.
            logger.exception("scoring job %s failed", probe.id)
            probe.summary = dict(
                job_id=probe.id, status="failed", error="the scorer failed for a reason unrelated to the corpus"
            )
            # The next run gets a fresh VM rather than one that may be broken.
            await self._drop_vm()
        finally:
            probe.job.cleanup()
            if self._vm is not None and self._release is None:
                self._release = asyncio.create_task(self._release_later())
        probe.summary["files"] = f"scores/{probe.id}/"
        (directory / "summary.json").write_text(json.dumps(probe.summary, indent=2))
        published = directory / "published"
        published.mkdir()
        (published / "summary.json").write_text(json.dumps(probe.summary, indent=2))
        if (directory / "out" / "metrics.jsonl").exists():
            records = [json.loads(line) for line in (directory / "out" / "metrics.jsonl").read_text().splitlines()]
            (published / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in public_metrics(records)))
        try:
            await probe.workspace.put_tree(published, f"scores/{probe.id}")
        except Exception:
            # `score_status` still returns the summary.
            logger.exception("could not copy job %s results into the workspace", probe.id)

    @vf.tool
    async def score(self, train_tokens: int | None = None) -> dict:
        """Train the model on a snapshot of the workspace's `corpus/` and measure its loss on the
        public part of the held-out evaluation set.

        The job runs in the background and its id is returned at once; later changes to `corpus/`
        do not affect it. `train_tokens` shortens the run (default: the full training budget).
        One job runs at a time. When it finishes, its summary and per-step metrics are written to
        `scores/<job_id>/` in the workspace.
        """
        budget, minimum = self.scoring.train_tokens, self.scoring.min_probe_tokens or self.batch_tokens
        tokens = budget if train_tokens is None else train_tokens
        if not minimum <= tokens <= budget:
            return dict(error=f"train_tokens must be between {minimum:,} and {budget:,}")
        async with self._submitting:
            running = [probe.id for probe in self.probes.values() if probe.summary is None]
            if running:
                return dict(error=f"job {running[0]} is still running; wait for it or cancel it")
            workspace = self._workspace()
            if not await workspace.exists("corpus"):
                return dict(error="corpus does not exist")
            probe_id = uuid.uuid4().hex[:12]
            directory = self.host_dir / "rollouts" / self.state.rollout / "probes" / probe_id
            await workspace.get_tree("corpus", directory)
            probe = Probe(probe_id, Job(self.scoring, self.host_dir, directory, "probe", tokens), workspace)
            probe.task = asyncio.create_task(self._run(probe))
            self.probes[probe_id] = probe
        steps = self._steps(tokens)
        return dict(job_id=probe_id, status="running", train_tokens=steps * self.batch_tokens, total_steps=steps)

    @vf.tool
    async def score_status(self, job_id: str, wait_seconds: float = 0.0) -> dict:
        """Report a scoring job's progress, or its results once it has finished.

        Waits up to `wait_seconds` (at most 3600) for the job to finish first.
        """
        probe = self.probes.get(job_id)
        if probe is None:
            return dict(error=f"no job {job_id!r}")
        if probe.summary is None and wait_seconds > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(probe.task), min(wait_seconds, MAX_WAIT_SECONDS))
        if probe.summary is not None:
            return probe.summary
        return dict(
            job_id=job_id,
            status="running",
            elapsed_seconds=round(time.time() - probe.submitted),
            total_steps=self._steps(probe.job.train_tokens),
            **probe.job.progress(),
        )

    @vf.tool
    async def cancel_score(self, job_id: str) -> dict:
        """Stop a running scoring job."""
        probe = self.probes.get(job_id)
        if probe is None:
            return dict(error=f"no job {job_id!r}")
        if probe.summary is not None:
            return dict(job_id=job_id, status=probe.summary["status"])
        probe.task.cancel()
        await asyncio.gather(probe.task, return_exceptions=True)
        probe.summary = dict(job_id=job_id, status="cancelled")
        return probe.summary


def probe_summaries(host_dir: Path, rollout: str) -> list[dict]:
    """Every finished probe of a rollout, in submission order."""
    paths = (host_dir / "rollouts" / rollout / "probes").glob("*/summary.json")
    return [json.loads(path.read_text()) for path in sorted(paths, key=lambda path: path.stat().st_mtime)]


if __name__ == "__main__":
    ScorerToolset.run()
