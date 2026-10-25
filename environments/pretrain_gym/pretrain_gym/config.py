"""Configuration: the training budget, model, held-out evaluation set, scorer, Hub access, and reward."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator
from pydantic_config import BaseConfig

MODELS_DIR = Path(__file__).parent / "models"
PALOMA_SUBSETS = [
    "c4_en",
    "mc4",
    "c4_100_domains",
    "falcon-refinedweb",
    "redpajama",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "wikitext_103",
    "ptb",
]


class EvalSetConfig(BaseConfig):
    """Held-out evaluation text: files of one Hugging Face dataset, by split and optionally by subset."""

    repo_id: str = "allenai/paloma"
    revision: str = "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"
    local_dir: Path | None = None
    """A host directory with the same layout, used instead of downloading the dataset."""
    token_env: str = "HF_TOKEN"
    """Host environment variable with the Hub token that downloads the evaluation set."""
    subsets: list[str] = Field(default_factory=lambda: list(PALOMA_SUBSETS))
    """Subsets whose losses are averaged; empty when the set has none and is scored as a whole."""
    files: str = "{subset}/{split}/*.jsonl.gz"
    """Glob, relative to the repository root, of one split's files (of one subset, if any)."""
    text_field: str = "text"
    val_split: str = "val"
    """Measured by the agent's scorer tool and removed from training data."""
    test_split: str = "test"
    """Measured by final scoring; removed from training data at final scoring."""

    @model_validator(mode="after")
    def _check_files(self) -> EvalSetConfig:
        if "{split}" not in self.files:
            raise ValueError("eval_set.files must contain {split}")
        if bool(self.subsets) != ("{subset}" in self.files):
            raise ValueError("eval_set.files must contain {subset} exactly when eval_set.subsets is set")
        return self


class ScorerConfig(BaseConfig):
    backend: Literal["host", "prime"] = "host"
    """Run scoring jobs on this host's GPUs, or on a Prime GPU VM started for them."""
    python: str | None = None
    """The Python that runs jobs; None uses this environment's on the host and `python` on a VM."""
    gpus: list[int] | None = None
    """Host backend: GPU indices that jobs lease one at a time; None uses every GPU."""
    workers: int | None = Field(None, ge=1)
    """Processes that read, decontaminate, and tokenize the corpus; None uses every CPU."""
    prime: vf.PrimeConfig = vf.PrimeConfig(
        image="pretrain-gym-scorer:v1", gpu="H200_141GB:1", cpu=16.0, memory=64.0, disk=100.0, idle_timeout=None
    )
    """Prime backend: the scorer VM. `image` must be in the Prime registry."""
    idle_release_seconds: float = Field(600.0, ge=0.0)
    """Prime backend: the scorer VM that serves the agent's runs is deleted after this long without one."""


class ScoringConfig(BaseConfig):
    train_tokens: int = Field(400_000_000, ge=1)
    """Tokens the model trains on at final scoring."""
    min_probe_tokens: int | None = Field(None, ge=1)
    """Smallest probe run; None allows one optimizer step of the model."""
    model: str = "nanogpt_speedrun"
    """A model directory bundled with this package, or a path to one."""
    eval_set: EvalSetConfig = EvalSetConfig()
    scorer: ScorerConfig = ScorerConfig()
    ngram: int = Field(13, ge=1)
    """Training text sharing this many consecutive words with held-out text is removed."""
    target_loss: float = Field(0.0, ge=0.0)
    """The loss at which performance reaches 1."""
    reward_exponent: float = Field(2.0, gt=0.0)
    export_weights: bool = False
    """Final scoring also saves the trained weights to `<host_dir>/rollouts/<rollout>/final/out/model.pt`."""

    @property
    def model_dir(self) -> Path:
        path = Path(self.model).expanduser()
        return path if path.is_absolute() else MODELS_DIR / self.model

    @model_validator(mode="after")
    def _check(self) -> ScoringConfig:
        if not (self.model_dir / "model.json").is_file() or not (self.model_dir / "train.py").is_file():
            raise ValueError(f"model {self.model!r} needs model.json and train.py in {self.model_dir}")
        if self.min_probe_tokens is not None and self.min_probe_tokens > self.train_tokens:
            raise ValueError("min_probe_tokens cannot exceed train_tokens")
        return self

    def performance(self, loss: float, vocab_size: int) -> float:
        """`p ** exponent` (or `p` below 0) for `p = (ln V - loss) / (ln V - target_loss)`."""
        if not math.isfinite(loss):
            return 0.0
        uniform = math.log(vocab_size)
        p = (uniform - loss) / (uniform - self.target_loss)
        return p**self.reward_exponent if p >= 0 else p


class HubConfig(BaseConfig):
    cutoff_date: date = date(2025, 12, 31)
    """Datasets must be created, and used at a commit made, on or before this day; papers published by it."""
    max_download_gb: float | None = Field(None, gt=0.0)
    """Total Hub data one rollout may download into its workspace; None is unlimited."""
    token_env: str = "HF_TOKEN"
    """Host environment variable with the Hub token the agent's Hub tools use."""


class WorkspaceConfig(BaseConfig):
    image: str = "pretrain-gym-agent:v1"
    cpu: float = 16.0
    memory: float = 64.0
    disk: float = 500.0


class PretrainTaskConfig(vf.TaskConfig):
    host_dir: Path = Path("~/.cache/pretrain_gym")
    """Host directory for evaluation data, caches, downloads in transit, and scoring jobs."""
    scoring: ScoringConfig = ScoringConfig()
    hub: HubConfig = HubConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    max_turns: int | None = None
    """The agent's turn limit, stated in the prompt; `CurationEnv` copies it from `env.agent.max_turns`."""


class PretrainTasksetConfig(vf.TasksetConfig):
    task: PretrainTaskConfig = PretrainTaskConfig()


class HubToolsetConfig(vf.ToolsetConfig):
    hub: HubConfig = HubConfig()
    host_dir: Path = Path("~/.cache/pretrain_gym")


class ScorerToolsetConfig(vf.ToolsetConfig):
    scoring: ScoringConfig = ScoringConfig()
    host_dir: Path = Path("~/.cache/pretrain_gym")
