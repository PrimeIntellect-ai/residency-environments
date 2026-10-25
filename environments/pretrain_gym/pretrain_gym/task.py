"""The curation task: workspace setup, final scoring on the test split, reward, and metrics."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import verifiers.v1 as vf
from prime_sandboxes import AsyncSandboxClient
from tokenizers import Tokenizer
from verifiers.v1.runtimes import DockerRuntime, PrimeRuntime

from .config import HubToolsetConfig, PretrainTaskConfig, PretrainTasksetConfig, ScorerToolsetConfig
from .hub import HubToolset
from .prompt import render
from .scorer import ScorerToolset, probe_summaries
from .scoring import CorpusError, Job, ScorerVM, model_spec, stage_code, tokenizer_file
from .state import PretrainState
from .workspace import WORKDIR, DockerBox, PrimeBox, open_box

TURNS_SCRIPT = Path(__file__).parent / "assets" / "turns.py"
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def agent_workspace(state: PretrainState) -> AsyncIterator[DockerBox | PrimeBox]:
    """The agent's workspace, with a Prime client open while it is in use."""
    if state.workspace_kind != "prime":
        yield open_box("docker", state.workspace)
        return
    client = AsyncSandboxClient()
    try:
        yield open_box("prime", state.workspace, client)
    finally:
        await client.aclose()


class PretrainTask(vf.Task[vf.TaskData, PretrainState, PretrainTaskConfig]):
    NEEDS_CONTAINER = True

    @classmethod
    def toolsets(cls, config: PretrainTaskConfig) -> list[vf.Toolset]:
        return [
            HubToolset(HubToolsetConfig(hub=config.hub, host_dir=config.host_dir)),
            ScorerToolset(ScorerToolsetConfig(scoring=config.scoring, host_dir=config.host_dir)),
        ]

    @property
    def host_dir(self):
        return self.config.host_dir.expanduser()

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if not isinstance(runtime, DockerRuntime | PrimeRuntime):
            raise vf.TaskError(f"pretrain-gym needs the docker or prime runtime, not {runtime.type}")
        state = cast(PretrainState, trace.state)
        state.workspace, state.workspace_kind, state.rollout = runtime.info.id, runtime.type, trace.id
        trainer = self.host_dir / "rollouts" / trace.id / "trainer"
        stage_code(self.config.scoring, trainer)
        for path in sorted(trainer.iterdir()):
            await runtime.write(f"{WORKDIR}/trainer/{path.name}", path.read_bytes())
        await runtime.write(f"{WORKDIR}/turns.py", TURNS_SCRIPT.read_bytes())
        await runtime.run(["mkdir", "-p", f"{WORKDIR}/corpus", f"{WORKDIR}/data"], {})

    @vf.stop
    async def record_turn(self, trace: vf.Trace) -> bool:
        """Write the turn about to start into the workspace for `turns.py`; never stops the run."""
        turn = json.dumps(dict(turn=trace.num_turns + 1, max_turns=trace.agent.config.max_turns))
        try:
            async with agent_workspace(cast(PretrainState, trace.state)) as workspace:
                await workspace.sh(f"printf %s {shlex.quote(turn)} > .turns.json")
        except Exception:
            # A stop hook that raises ends the rollout; a stale turn count should not.
            logger.warning("could not record turn %d in the workspace", trace.num_turns + 1, exc_info=True)
        return False

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        state = cast(PretrainState, trace.state)
        scoring = self.config.scoring
        directory = self.host_dir / "rollouts" / trace.id / "final"
        async with agent_workspace(state) as workspace:
            if not await workspace.exists("corpus"):
                state.result = dict(corpus_error="corpus does not exist")
                return
            await workspace.get_tree("corpus", directory)
        job = Job(scoring, self.host_dir, directory, "final", scoring.train_tokens)
        vm = ScorerVM(scoring.scorer.prime) if scoring.scorer.backend == "prime" else None
        try:
            if vm is not None:
                await vm.start()
            result = await job.run(vm)
        except CorpusError as exc:
            state.result = dict(corpus_error=str(exc))
            return
        finally:
            job.cleanup()
            if vm is not None:
                await vm.stop()
        state.result = dict(
            loss=result.loss,
            bits_per_byte=result.bpb,
            vocab_size=result.vocab_size,
            losses=result.losses,
            bits_per_byte_by_set=result.bits_per_byte,
            train=result.train,
            prepare=result.prepare,
        )
        (directory / "result.json").write_text(json.dumps(state.result, indent=2))

    @vf.reward
    async def performance(self, trace: vf.Trace) -> float:
        result = cast(PretrainState, trace.state).result
        if "loss" not in result:
            return 0.0
        return self.config.scoring.performance(result["loss"], result["vocab_size"])

    @vf.metric
    async def scoring(self, trace: vf.Trace) -> dict[str, float]:
        state = cast(PretrainState, trace.state)
        result = state.result
        metrics: dict[str, float] = dict(corpus_error=float("corpus_error" in result))
        if "loss" in result:
            prepare, train = result["prepare"], result["train"]
            metrics.update(
                loss=result["loss"],
                bits_per_byte=result["bits_per_byte"],
                diverged=float(train["diverged"]),
                train_tokens=train["train_tokens"],
                train_seconds=train["train_seconds"],
                corpus_documents=prepare["documents"],
                corpus_short_documents=prepare["documents_short"],
                heldout_documents_changed=prepare["documents_changed"],
                heldout_characters_removed=prepare["characters_removed"],
                heldout_character_fraction=prepare["characters_removed"] / max(1, prepare["characters"]),
                corpus_exhausted=float(prepare["exhausted"]),
                **{f"loss/{name}": loss for name, loss in result["losses"].items()},
                **{f"bits_per_byte/{name}": bpb for name, bpb in result["bits_per_byte_by_set"].items()},
            )
        probes = probe_summaries(self.host_dir, state.rollout)
        done = [probe for probe in probes if probe["status"] == "done"]
        metrics.update(
            probes=len(probes),
            probes_failed=sum(probe["status"] == "failed" for probe in probes),
            probe_tokens=sum(probe["train_tokens"] for probe in done),
        )
        if done:
            metrics.update(best_probe_loss=min(p["val_loss"] for p in done), last_probe_loss=done[-1]["val_loss"])
        downloads = self.host_dir / "rollouts" / state.rollout / "downloads.jsonl"
        records = [json.loads(line) for line in downloads.read_text().splitlines()] if downloads.exists() else []
        metrics.update(downloads=len(records), download_bytes=sum(record["bytes"] for record in records))
        return {key: float(value) for key, value in metrics.items() if math.isfinite(value)}


class PretrainTaskset(vf.Taskset[PretrainTask, PretrainTasksetConfig]):
    def load(self) -> list[PretrainTask]:
        config = self.config.task
        model = model_spec(config.scoring)
        vocab_size = Tokenizer.from_file(str(tokenizer_file(model))).get_vocab_size()
        workspace = config.workspace
        data = vf.TaskData(
            prompt=render(config, vocab_size),
            image=workspace.image,
            workdir=WORKDIR,
            network_allow=[],
            resources=vf.TaskResources(cpu=workspace.cpu, memory=workspace.memory, disk=workspace.disk),
        )
        return [PretrainTask(data, config)]
