"""Scoring jobs: train the configured model on a corpus snapshot and measure held-out loss.

A job directory on the host holds the snapshot (`corpus/`), the staged runner and model code
(`code/`), the token files (`data/`), and the trainer's outputs (`out/`). On the host backend,
preparation runs without a GPU and training leases one host GPU at a time across every process on
the host. On the Prime backend, the job directory is uploaded to a GPU VM, runs there, and its
outputs are copied back.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import fnmatch
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from prime_sandboxes import AsyncSandboxClient
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from verifiers.v1.runtimes import make_runtime

from .config import ScoringConfig
from .workspace import PrimeBox

RUNNER_DIR = Path(__file__).parent / "runner"
Purpose = Literal["probe", "final"]
GPU_POLL_SECONDS = 5.0
PROGRESS_SECONDS = 30.0


class CorpusError(Exception):
    """The agent's corpus cannot be read or yields no training text."""


class ScoringError(Exception):
    """A scoring job failed for a reason other than the corpus."""


@dataclass
class JobResult:
    losses: dict[str, float]
    bits_per_byte: dict[str, float]
    vocab_size: int
    train: dict
    prepare: dict
    eval_sets: dict[str, dict] = field(default_factory=dict)

    @property
    def loss(self) -> float:
        return statistics.fmean(self.losses.values()) if self.losses else float("inf")

    @property
    def bpb(self) -> float:
        return statistics.fmean(self.bits_per_byte.values()) if self.bits_per_byte else float("inf")


def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("w")
    fcntl.flock(file, fcntl.LOCK_EX)
    return file


# Large gated snapshots hit Hub rate limits partway; each retry resumes with the files still missing.
_download_snapshot = retry(
    retry=retry_if_not_exception_type((GatedRepoError, RepositoryNotFoundError)),
    stop=stop_after_attempt(10),
    wait=wait_exponential(min=10, max=300),
    reraise=True,
)(snapshot_download)


def eval_parts(config: ScoringConfig) -> list[str]:
    """The evaluation set's subsets, or a single part named `all` when it has none."""
    return config.eval_set.subsets or ["all"]


def eval_root(config: ScoringConfig, host_dir: Path) -> Path:
    eval_set = config.eval_set
    if eval_set.local_dir is not None:
        return eval_set.local_dir.expanduser().resolve()
    return host_dir / "eval_sets" / eval_set.repo_id / eval_set.revision


def eval_files(config: ScoringConfig, host_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """The evaluation set's files by (part, split), downloaded once per host."""
    eval_set = config.eval_set
    root = eval_root(config, host_dir)
    splits = (eval_set.val_split, eval_set.test_split)
    parts = eval_parts(config)
    patterns = [eval_set.files.format(subset=part, split=split) for part in parts for split in splits]
    if eval_set.local_dir is None:
        with _locked(root.with_suffix(".lock")):
            if not (root / ".complete").exists():
                _download_snapshot(
                    eval_set.repo_id,
                    repo_type="dataset",
                    revision=eval_set.revision,
                    allow_patterns=patterns,
                    local_dir=root,
                    max_workers=2,
                    token=os.environ.get(eval_set.token_env),
                )
                shutil.rmtree(root / ".cache", ignore_errors=True)
                (root / ".complete").touch()
    found = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    files = {}
    for part in parts:
        for split in splits:
            pattern = eval_set.files.format(subset=part, split=split)
            matched = [root / path for path in found if fnmatch.fnmatch(path, pattern)]
            if not matched:
                raise ScoringError(f"evaluation set {eval_set.repo_id} has no files matching {pattern!r}")
            files[part, split] = matched
    return files


def tokenizer_file(model: dict) -> Path:
    tokenizer = model["tokenizer"]
    return Path(hf_hub_download(tokenizer["repo_id"], "tokenizer.json", revision=tokenizer["revision"]))


def model_spec(config: ScoringConfig) -> dict:
    return json.loads((config.model_dir / "model.json").read_text())


def stage_code(config: ScoringConfig, destination: Path) -> None:
    """The runner, the model's files, and its tokenizer: everything the agent's `trainer/` shows too."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in (RUNNER_DIR, config.model_dir):
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                shutil.copy2(path, destination / path.name)
    shutil.copy2(tokenizer_file(model_spec(config)), destination / "tokenizer.json")


@contextlib.asynccontextmanager
async def gpu_lease(gpus: list[int] | None, lock_dir: Path) -> AsyncIterator[int]:
    """Hold one host GPU exclusively among every process that leases from `lock_dir`."""
    if gpus is None:
        listed = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True)
        gpus = list(range(len(listed.stdout.strip().splitlines())))
    if not gpus:
        raise ScoringError("no GPUs to lease")
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for gpu in gpus:
            file = (lock_dir / f"gpu{gpu}.lock").open("w")
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                file.close()
                continue
            try:
                yield gpu
            finally:
                file.close()
            return
        await asyncio.sleep(GPU_POLL_SECONDS)


class ScorerVM:
    """A Prime GPU VM for scoring jobs; the evaluation set is uploaded once, and caches persist between jobs."""

    def __init__(self, config: vf.PrimeConfig) -> None:
        self.runtime = make_runtime(config)
        self.client = AsyncSandboxClient()
        self.box: PrimeBox | None = None
        self.has_eval_set = False

    async def start(self) -> None:
        await self.runtime.start()
        self.box = PrimeBox(self.runtime.info.id, self.client, root="/")

    async def stop(self) -> None:
        try:
            await self.runtime.stop()
        finally:
            await self.client.aclose()

    async def stage(self, directory: Path, eval_root: Path) -> None:
        if not self.has_eval_set:
            await self.box.put_tree(eval_root, "evalset")
            self.has_eval_set = True
        await self.box.sh("rm -rf /job")
        await self.box.put_tree(directory, "job")


class Job:
    """One scoring run over the corpus snapshot in `<directory>/corpus`, on the host or on a scorer VM."""

    def __init__(self, config: ScoringConfig, host_dir: Path, directory: Path, purpose: Purpose, train_tokens: int):
        self.config, self.host_dir, self.directory = config, host_dir, directory
        self.purpose, self.train_tokens = purpose, train_tokens
        self.stage = "queued"
        self.vm: ScorerVM | None = None

    def _path(self, host_path: Path, remote: str) -> str:
        return remote if self.vm is not None else str(host_path)

    def _spec(self, files: dict[tuple[str, str], list[Path]], root: Path) -> dict:
        eval_set = self.config.eval_set
        split = eval_set.val_split if self.purpose == "probe" else eval_set.test_split
        screened = [eval_set.val_split] if self.purpose == "probe" else [eval_set.val_split, eval_set.test_split]
        parts = eval_parts(self.config)

        def eval_path(path: Path) -> str:
            return self._path(path, f"/evalset/{path.relative_to(root).as_posix()}")

        job, cache = self.directory, self.host_dir / "cache"
        return dict(
            corpus=self._path(job / "corpus", "/job/corpus"),
            tokenizer=self._path(job / "code" / "tokenizer.json", "/job/code/tokenizer.json"),
            bos_token=model_spec(self.config)["tokenizer"]["bos_token"],
            train_tokens=self.train_tokens,
            eval={part: [eval_path(path) for path in files[part, split]] for part in parts},
            screen=[eval_path(path) for part in parts for s in screened for path in files[part, s]],
            ngram=self.config.ngram,
            text_field=eval_set.text_field,
            cache=self._path(cache, "/cache"),
            out=self._path(job / "data", "/job/data"),
            workers=self.config.scorer.workers,
        )

    async def _run(self, argv: list[str], gpu: int | None = None) -> None:
        cache, log = self.host_dir / "cache", self.directory / "job.log"
        env = {
            "PYTHONHASHSEED": "0",
            "TORCHINDUCTOR_CACHE_DIR": self._path(cache / "inductor", "/cache/inductor"),
            "TRITON_CACHE_DIR": self._path(cache / "triton", "/cache/triton"),
        }
        if self.vm is not None:
            python = self.config.scorer.python or "python"
            try:
                result = await self.vm.runtime.run([python, *argv], env)
            except BaseException:
                # The VM keeps running a command whose caller is cancelled.
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.vm.box.sh("pkill -f /job/code/ || true"))
                raise
            output = result.stdout + result.stderr
            with log.open("a") as file:
                file.write(output)
            if result.exit_code:
                raise ScoringError(f"{Path(argv[0]).name} exited with {result.exit_code}:\n{output[-4000:]}")
            return
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        with log.open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                self.config.scorer.python or sys.executable,
                *argv,
                stdout=output,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
            )
            try:
                code = await process.wait()
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                raise
        if code:
            tail = log.read_bytes()[-4000:].decode(errors="replace")
            raise ScoringError(f"{Path(argv[0]).name} exited with {code}:\n{tail}")

    async def _fetch(self, *names: str) -> None:
        """Copy job outputs from the scorer VM to the host job directory."""
        if self.vm is None:
            return
        for name in names:
            if await self.vm.box.exists(f"/job/{name}"):
                await self.vm.box.get_file(f"/job/{name}", self.directory / name)

    async def _train(self, argv: list[str]) -> None:
        if self.vm is None:
            self.stage = "waiting for a GPU"
            async with gpu_lease(self.config.scorer.gpus, self.host_dir / "gpu_locks") as gpu:
                self.stage = "training"
                await self._run(argv, gpu)
            return
        self.stage = "training"
        training = asyncio.create_task(self._run(argv))
        try:
            while not training.done():
                await asyncio.wait({training}, timeout=PROGRESS_SECONDS)
                # Progress only; a failed copy is retried at the next poll.
                with contextlib.suppress(Exception):
                    await self._fetch("out/metrics.jsonl")
        finally:
            if not training.done():
                training.cancel()
            await asyncio.gather(training, return_exceptions=True)
        await training

    async def run(self, vm: ScorerVM | None = None) -> JobResult:
        self.vm = vm
        code, data, out = self.directory / "code", self.directory / "data", self.directory / "out"
        (self.host_dir / "cache").mkdir(parents=True, exist_ok=True)
        self.stage = "preparing"
        files = await asyncio.to_thread(eval_files, self.config, self.host_dir)
        root = eval_root(self.config, self.host_dir)
        await asyncio.to_thread(stage_code, self.config, code)
        (self.directory / "spec.json").write_text(json.dumps(self._spec(files, root), indent=2))
        if vm is not None:
            await vm.stage(self.directory, root)
        await self._run(
            [
                self._path(code / "prepare.py", "/job/code/prepare.py"),
                self._path(self.directory / "spec.json", "/job/spec.json"),
            ]
        )
        await self._fetch("data/meta.json", "data/prepare.json", "data/corpus_error.json")
        if (data / "corpus_error.json").exists():
            raise CorpusError(json.loads((data / "corpus_error.json").read_text())["error"])
        train = [
            self._path(code / "train.py", "/job/code/train.py"),
            "--config",
            self._path(code / "model.json", "/job/code/model.json"),
            "--data",
            self._path(data, "/job/data"),
            "--out",
            self._path(out, "/job/out"),
            "--train-tokens",
            str(self.train_tokens),
        ]
        export = self.purpose == "final" and self.config.export_weights
        if export:
            train += ["--save-weights", self._path(out / "model.pt", "/job/out/model.pt")]
        await self._train(train)
        await self._fetch("out/result.json", "out/metrics.jsonl", *(["out/model.pt"] if export else []))
        self.stage = "done"
        meta = json.loads((data / "meta.json").read_text())
        result = json.loads((out / "result.json").read_text())
        return JobResult(
            losses=result.pop("losses"),
            bits_per_byte=result.pop("bits_per_byte"),
            vocab_size=meta["vocab_size"],
            train=result,
            prepare=json.loads((data / "prepare.json").read_text()),
            eval_sets=meta["eval"],
        )

    def progress(self) -> dict:
        """The job's stage and, once training starts, its latest step."""
        progress = dict(stage=self.stage)
        metrics = self.directory / "out" / "metrics.jsonl"
        if metrics.exists():
            records = [json.loads(line) for line in metrics.read_text().splitlines()]
            steps = [record for record in records if "train_loss" in record]
            if steps:
                progress.update(step=steps[-1]["step"], tokens=steps[-1]["tokens"], train_loss=steps[-1]["train_loss"])
            if records and records[-1].get("evaluating"):
                progress.update(stage="evaluating")
        return progress

    def cleanup(self) -> None:
        """Delete the snapshot and token files; keep the specification, log, and outputs."""
        for name in ("corpus", "data"):
            shutil.rmtree(self.directory / name, ignore_errors=True)
