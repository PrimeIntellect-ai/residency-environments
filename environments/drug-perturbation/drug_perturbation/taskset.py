"""Full-chain and optional intermediate drug perturbation prediction tasks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

import pandas as pd
import verifiers.v1 as vf
from huggingface_hub import hf_hub_download
from pydantic import Field, model_validator

from .judge import ProcessJudgeConfig, judge_process
from .process_contract import TASK_GRAPHS
from .scoring import DEFAULT_REWARD_WEIGHTS, parse_answer, score_response
from .tools import CompoundToolset

DATASET_ID = "seanpohorence/drug-perturbation"
DATASET_REVISION = "49d20f9f293fea3f4c93ab57b478aaa3f1651d84"
DATASET_SHA256 = "9f1cedc98d93f84069d4ed823325adb6e03fca26c611ca42e11628d057b9c048"
SPLIT_SHA256 = {
    "train": "a791ea9b1b4ad21dcd62241159f173727efafdfd8e57df5b233d04d054e169a1",
    "test": "933192b6106d8c449aca84b6598cdf8046cc93f76082dd84d1a289d7e98552b0",
}

EntryPoint = Literal[
    "smiles_only",
    "from_target",
    "from_moa",
    "from_pathways",
    "phenotype_direct",
    "target_from_smiles",
    "moa_from_smiles",
    "moa_from_target",
    "pathways_from_smiles",
    "pathways_from_moa",
    "phenotype_from_moa",
    "phenotype_from_pathways",
]
Phenotype = Literal["viability", "cell_cycle", "stress", "magnitude"]


class DrugPerturbationTaskData(vf.TaskData):
    answer: str
    compound: str
    cell_line: str
    entry_point: EntryPoint
    phenotype: str
    source_key: str
    prompt_key: str
    source_row: int


class DrugPerturbationTaskConfig(vf.TaskConfig):
    component_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    tools_enabled: bool = True
    compound_tools: vf.ToolsetConfig = Field(default_factory=vf.ToolsetConfig)
    process_judge: ProcessJudgeConfig = Field(default_factory=ProcessJudgeConfig)

    def reward_weight(self, name: str) -> float:
        default = 1.0 if name == "deterministic_reward" else 0.0
        override = self.rewards.get(name)
        return override.weight if override is not None and override.weight is not None else default

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        values = [self.reward_weight(name) for name in ("deterministic_reward", "process_adjusted_reward")]
        if any(value < 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError(
                "deterministic_reward and process_adjusted_reward weights must be nonnegative and sum to one"
            )
        if set(self.component_weights) != set(DEFAULT_REWARD_WEIGHTS):
            raise ValueError("component_weights must contain exactly target, moa, pathways, and phenotype")
        if any(not math.isfinite(value) or value < 0 for value in self.component_weights.values()):
            raise ValueError("component_weights must be finite and nonnegative")
        if not sum(self.component_weights.values()):
            raise ValueError("At least one biological component weight must be positive")
        if self.process_judge.prompt is not None:
            raise ValueError("The versioned process rubric does not support a prompt-file override")
        return self


class DrugPerturbationTask(vf.Task[DrugPerturbationTaskData, vf.State, DrugPerturbationTaskConfig]):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.source_key

    @classmethod
    def toolsets(cls, config: DrugPerturbationTaskConfig) -> list[vf.Toolset]:
        return [CompoundToolset(config.compound_tools)] if config.tools_enabled else []

    def _scores(self, trace: vf.Trace) -> dict[str, float]:
        return score_response(trace.last_reply, self.data.answer, self.config.component_weights)

    @vf.metric
    async def biological_metrics(self, trace: vf.Trace) -> dict[str, float]:
        scores = self._scores(trace)
        answer = parse_answer(self.data.answer)
        requested = {
            "target": bool(answer.get("target")),
            "moa": bool(answer.get("moa")),
            "pathways": bool(answer.get("pathways_signed")),
            "phenotype": bool(answer.get("phenotype")),
        }
        metrics = {f"{name}_applicable": float(value) for name, value in requested.items()}
        metrics["deterministic_reward"] = scores["aggregate_reward"]
        metrics["format_compliance"] = scores["format_compliance"]
        for name, value in scores.items():
            if name in ("aggregate_reward", "format_compliance"):
                continue
            component = "pathways" if name.startswith("pathway_") else name.split("_")[0]
            if requested[component]:
                metrics[name] = value
            else:
                trace.metrics[name] = None
        return metrics

    @vf.reward(weight=1.0)
    async def deterministic_reward(self, trace: vf.Trace) -> float:
        return self._scores(trace)["aggregate_reward"]

    @vf.reward(weight=0.0)
    async def process_adjusted_reward(self, trace: vf.Trace) -> dict[str, float]:
        if self.config.reward_weight("process_adjusted_reward") == 0:
            trace.record_metric("process_judge_computed", 0.0)
            trace.metrics["process_score"] = None
            trace.metrics["process_adjusted_reward"] = None
            return {}
        metrics = await judge_process(trace, self.data.entry_point, self.config.process_judge)
        deterministic = self._scores(trace)["aggregate_reward"]
        adjusted = deterministic * metrics["process_score"]
        trace.record_metrics(
            {
                **metrics,
                "process_adjusted_reward": adjusted,
                "process_attenuation": deterministic - adjusted,
                "positive_reward_attenuated": float(deterministic > 0 and adjusted < deterministic),
            }
        )
        return {"process_adjusted_reward": adjusted}


class DrugPerturbationTasksetConfig(vf.TasksetConfig):
    task: DrugPerturbationTaskConfig = Field(default_factory=DrugPerturbationTaskConfig)
    split: Literal["train", "test"] = "test"
    entry_points: list[EntryPoint] = Field(default_factory=lambda: ["smiles_only"])
    phenotypes: list[Phenotype] = Field(default_factory=lambda: ["viability", "cell_cycle", "stress"])
    cell_lines: list[str] | None = None
    seed: int = 42
    dataset_path: Path | None = None
    """Optional local copy of the exact, checksum-validated published Parquet file."""

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.entry_points or not self.phenotypes or self.cell_lines == []:
            raise ValueError("Task and phenotype selections must be nonempty")
        if len(self.entry_points) != len(set(self.entry_points)):
            raise ValueError("entry_points must not contain duplicates")
        if self.task.reward_weight("process_adjusted_reward") > 0:
            unsupported = set(self.entry_points) - set(TASK_GRAPHS)
            if unsupported or "magnitude" in self.phenotypes:
                raise ValueError(
                    "The optional process rubric supports smiles_only and the four direct links, "
                    "with viability/cell_cycle/stress. Other views require deterministic-only scoring."
                )
        return self


class DrugPerturbationTaskset(vf.Taskset[DrugPerturbationTask, DrugPerturbationTasksetConfig]):
    def load(self) -> Iterator[DrugPerturbationTask]:
        path = self.config.dataset_path
        expected_hash = DATASET_SHA256
        if path is None:
            path = Path(
                hf_hub_download(
                    DATASET_ID, f"data/{self.config.split}.parquet", repo_type="dataset", revision=DATASET_REVISION
                )
            )
            expected_hash = SPLIT_SHA256[self.config.split]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Dataset checksum mismatch: expected {expected_hash}, got {digest}")
        frame = pd.read_parquet(path)
        frame = frame[frame["split"] == self.config.split].reset_index(drop=True)
        frame["source_row"] = frame.index
        selected = frame["entry_point"].isin(self.config.entry_points)
        selected &= (frame["phenotype"] == "upstream") | frame["phenotype"].isin(self.config.phenotypes)
        if self.config.cell_lines is not None:
            selected &= frame["cell_line"].isin(self.config.cell_lines)
        frame = frame[selected].sample(frac=1, random_state=self.config.seed)
        if frame.empty:
            raise ValueError("No examples match the selected split/tasks/phenotypes/cell lines")
        seen: set[str] = set()
        for index, row in enumerate(frame.to_dict(orient="records")):
            identity = {
                key: row[key]
                for key in (
                    "split",
                    "source_row",
                    "compound",
                    "cell_line",
                    "entry_point",
                    "phenotype",
                    "user_prompt",
                    "answer_json",
                )
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                raise ValueError(f"Duplicate task key in the frozen dataset: {key}")
            seen.add(key)
            answer = parse_answer(row["answer_json"])
            applicable = [
                name
                for name, field in (
                    ("target", "target"),
                    ("moa", "moa"),
                    ("pathways", "pathways_signed"),
                    ("phenotype", "phenotype"),
                )
                if answer.get(field)
            ]
            if not sum(self.config.task.component_weights[name] for name in applicable):
                raise ValueError(f"All requested biological components have zero weight for task {key}")
            yield DrugPerturbationTask(
                DrugPerturbationTaskData(
                    idx=index,
                    name=f"{row['entry_point']}/{row['phenotype']}/{key[:12]}",
                    prompt=row["user_prompt"],
                    network_allow=[],
                    answer=row["answer_json"],
                    compound=row["compound"],
                    cell_line=row["cell_line"],
                    entry_point=row["entry_point"],
                    phenotype=row["phenotype"],
                    source_key=key,
                    prompt_key=hashlib.sha256(row["user_prompt"].encode()).hexdigest(),
                    source_row=row["source_row"],
                ),
                self.config.task,
            )
