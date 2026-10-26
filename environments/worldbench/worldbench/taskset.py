"""WorldBench task data, deterministic scoring, and Hugging Face loader."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, TypeAlias, cast

import verifiers.v1 as vf
from datasets import load_dataset

from worldbench.formatting import format_prompt, parse_answer
from worldbench.images import bounded_image_data_url

DATASET_ID = "zlab-princeton/WorldBench"
DATASET_REVISION = "c2d372ee38b4c2d486e0db8b9a35d24c7200998b"
DATASET_SPLIT = "train"

AnswerLabel: TypeAlias = Literal["A", "B", "C", "D"]


class WorldBenchData(vf.TaskData):
    question: str
    options: dict[AnswerLabel, str]
    answer: AnswerLabel
    category: str
    domain: str


class WorldBenchTask(vf.Task[WorldBenchData]):
    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float(bool(parse_answer(trace.last_reply)))

    @vf.reward
    async def correct_answer(self, trace: vf.Trace) -> float:
        return float(parse_answer(trace.last_reply) == self.data.answer)


class WorldBenchTaskset(vf.Taskset[WorldBenchTask, vf.TasksetConfig]):
    def load(self) -> Iterator[WorldBenchTask]:
        dataset = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split=DATASET_SPLIT,
            streaming=True,
        )
        for idx, row in enumerate(dataset):
            options: dict[AnswerLabel, str] = {
                "A": row["option_a"],
                "B": row["option_b"],
                "C": row["option_c"],
                "D": row["option_d"],
            }
            image_bytes = row["image"]["bytes"]
            image_url = bounded_image_data_url(image_bytes)

            prompt = [
                vf.UserMessage(
                    content=[
                        vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image_url)),
                        vf.TextContentPart(text=format_prompt(row["question"], options)),
                    ]
                )
            ]
            data = WorldBenchData(
                idx=idx,
                name=f"worldbench-{idx}",
                prompt=prompt,
                question=row["question"],
                options=options,
                answer=cast(AnswerLabel, row["answer"].strip().upper()),
                category=row["category"],
                domain=row["domain"],
            )
            yield WorldBenchTask(data, self.config.task)
