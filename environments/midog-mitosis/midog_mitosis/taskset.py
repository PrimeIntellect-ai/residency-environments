"""midog-mitosis taskset (Verifiers v1).

Verifiable single-turn vision tasks on MIDOG++ (CC0): find every mitotic figure in an H&E image and answer with the
coordinates of each. The tasks are frozen in two configs of the dataset: ``tasks``, the benchmark (at least three
figures per image, including 36 retained hard tasks), and ``warmup``, fields with one figure, for training small models.
A row holds the image shown, the exact question and the expected answer, so this module plans nothing.

The answer space is open (a list of points) and the reward is binary: 1 only if the listed points and the annotated
mitotic figures match one-to-one, none missing and none extra (see ``scoring.py``).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator

from .scoring import Evaluation, score_find

__all__ = ["MidogConfig", "MidogData", "MidogTask", "MidogTaskset", "final_reply"]

DEFAULT_DATASET = "tirandazdylan/midog-mitosis"
DATASET_REVISION = "d9616cb2d5207d9b06943c0a263dbd4025228bea"
"""Immutable dataset input; every evaluation must identify its dataset revision."""
SANDBOX_IMAGE = "prime/dtirandaz/midog-mitosis-sandbox:0.8.0"
"""Python 3.11 with Pillow 12.3.0 and NumPy 2.4.6 preinstalled (``scripts/midog-mitosis/sandbox/Dockerfile``), so a rollout
installs nothing. Prime image references take a tag, not a digest, and a tag can be pushed again, so this records the
registry build behind ``0.8.0`` for every reported run: image ``zmex6nh1qoljeb1kgxinfhdc``, build
``w24lzpop0w7jt9l4jtd0izi7``, pushed 2026-10-05. A new push of the tag gets a new build id, which the image's owner sees
in the ``fullImagePath`` of ``prime images list -o json``."""
WORKSPACE_NOTE = (
    "The image is also saved as image.jpg in the working directory. Python 3 has Pillow and NumPy installed. "
    "The sandbox has no network."
)
"""Added to every prompt: what the sandbox holds, nothing about how to use it."""
TIMEOUT = vf.TaskTimeout(setup=15 * 60, agent=2 * 60 * 60)
"""Applies only where the agent sets no timeout of its own (by default it sets none and waits forever): a hung setup or
agent then ends as an errored rollout instead of stalling the run."""


class MidogData(vf.TaskData):
    """One immutable task: an image and the annotated centers of every mitotic figure in it."""

    row_key: str
    """Source of the image: a field (``s0549_f03``) or two fields of one slide side by side (``s0549_f03+s0549_f05``)."""
    answer: str
    """The content of the expected ``Answer:`` line, as stored in the dataset."""
    width: int
    height: int
    targets: int
    """Number of mitotic figures in the image."""
    points: list[tuple[float, float]] = Field(min_length=1)
    """Mitotic-figure centers in image pixels."""

    @model_validator(mode="after")
    def _one_point_per_figure(self) -> MidogData:
        if self.targets != len(self.points):
            raise ValueError(f"{self.name}: {self.targets} figures but {len(self.points)} points")
        return self


def final_reply(trace: vf.Trace) -> str:
    """The agent's last reply in the conversation that began with the task prompt.

    A harness may call the model again after the answer for its own bookkeeping: Pi summarizes its context once it nears
    the window. Such a call is a separate conversation in the trace, so it is never taken for the agent's reply.
    """
    if not trace.nodes:
        return ""
    first = trace.nodes[0]
    main = [branch for branch in trace.branches if branch.nodes[0] is first]
    replies = [
        node.message for node in main[-1].nodes if node.sampled and isinstance(node.message, vf.AssistantMessage)
    ]
    return (replies[-1].content or "").strip() if replies else ""


class MidogTask(vf.Task[MidogData]):
    NEEDS_CONTAINER = True

    def image_bytes(self) -> bytes:
        """The JPEG shown to the model, decoded from the prompt's data URL."""
        url = next(p.image_url.url for p in self.data.prompt[0].content if isinstance(p, vf.ImageUrlContentPart))
        return base64.b64decode(url.partition(",")[2])

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write("image.jpg", self.image_bytes())

    @property
    def key(self) -> str:
        return self.data.row_key

    def evaluate(self, reply: str) -> Evaluation:
        data = self.data
        return score_find(reply, data.points, data.width, data.height)

    def gold_reply(self) -> str:
        return f"Answer: {self.data.answer}"

    @vf.reward(weight=1.0)
    async def answer_reward(self, trace: vf.Trace) -> float:
        """1 if the predicted centers match the annotated ones one-to-one, else 0."""
        return self.evaluate(final_reply(trace)).reward

    @vf.metric
    async def find_metrics(self, trace: vf.Trace) -> dict[str, float]:
        """How close the answer was: exact, F1, precision, recall, whether the reply parsed, points listed."""
        return {f"find/{key}": value for key, value in self.evaluate(final_reply(trace)).metrics.items()}

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The stored answer, phrased as a reply, earns full marks; the prompt carries a JPEG; setup staged that JPEG."""
        urls = [p.image_url.url for p in self.data.prompt[0].content if isinstance(p, vf.ImageUrlContentPart)]
        has_jpeg = len(urls) == 1 and urls[0].startswith("data:image/jpeg;base64,/9j/")
        staged = await runtime.read("image.jpg") == self.image_bytes()
        return self.evaluate(self.gold_reply()).reward == 1.0 and has_jpeg and staged


class MidogConfig(vf.TasksetConfig):
    dataset: str = DEFAULT_DATASET
    """HuggingFace dataset repo id, or a local directory produced by ``generators/midog-mitosis/build_dataset.py``."""
    revision: str | None = DATASET_REVISION
    """Dataset git revision (branch, tag, or commit); defaults to the immutable benchmark release."""
    config: Literal["tasks", "warmup"] = "tasks"
    """``tasks`` is the benchmark; ``warmup`` (one figure per field) is training material for small models."""
    split: Literal["train", "validation", "test"] = "test"
    """Custom slide-disjoint split; all train/validation configs exclude the test slides."""


def data_url(image: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image).decode()


class MidogTaskset(vf.Taskset[MidogTask, MidogConfig]):
    def load(self) -> Iterator[MidogTask]:
        from datasets import Dataset, Image, load_dataset

        config = self.config
        root = Path(config.dataset).expanduser()
        if root.is_dir():
            # Card-based local loads can share a cache key with another directory of the same name.
            files = sorted(str(path) for path in (root.resolve() / config.config).glob(f"{config.split}-*.parquet"))
            dataset = Dataset.from_parquet(files)
        else:
            dataset = load_dataset(config.dataset, config.config, split=config.split, revision=config.revision)
        dataset = dataset.sort("rank")
        rows = dataset.select_columns([c for c in dataset.column_names if c != "image"]).to_list()
        images = dataset.select_columns(["image"]).cast_column("image", Image(decode=False))
        for idx, row in enumerate(rows):
            message = vf.UserMessage(
                content=[
                    vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=data_url(images[idx]["image"]["bytes"]))),
                    vf.TextContentPart(text=row["question"]),
                    vf.TextContentPart(text=WORKSPACE_NOTE),
                ]
            )
            data = MidogData(
                idx=idx,
                name=row["task_id"],
                image=SANDBOX_IMAGE,
                network_allow=[],
                timeout=TIMEOUT,
                prompt=[message],
                row_key=row["source_id"],
                answer=row["answer"],
                width=row["width"],
                height=row["height"],
                targets=row["n_targets"],
                points=[tuple(point) for point in row["points"]],
            )
            yield MidogTask(data, config.task)
