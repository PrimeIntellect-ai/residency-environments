"""Periapical lesion taskset (Verifiers v1).

Each task is one dental pantomograph on a reporting worklist. The agent is the radiographer on
shift: it reads the referral, reports every periapical lesion with its Periapical Index grade per
the clinic's codebook, and files the report with the clinic's `radreport` tool. The reward scores
the filed report against the dentists' reference boxes and grades; a reply in chat is never read.
The film is in the prompt and in the sandbox workspace, and the sandbox has no network because
the reference labels are public.
"""

import asyncio
import base64
import hashlib
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import verifiers.v1 as vf
from datasets import Image, load_dataset
from scipy.optimize import linear_sum_assignment

__all__ = ["PeriapicalConfig", "PeriapicalData", "PeriapicalTask", "PeriapicalTaskset"]

DATASET = "tirandazdylan/periapical-lesions-pai"
REVISION = "3c74b963b449ae66ef23266a38812688dbe8cc81"

SANDBOX_IMAGE = "prime/dylantirandaz/periapical-lesions:v1"
"""Ships numpy and Pillow so the film can be analyzed with the network blocked."""

JPEG_DATA_URL = "data:image/jpeg;base64,"

MATCH_IOU = 0.3
"""Lesion boxes span tens of pixels on a ~2,450 px film, and the source dentists'
own re-annotations of one lesion sometimes overlap by less than 0.5."""

WORKLIST_PROMPT = (
    "One study is on today's worklist. It opens below and is saved as pantomograph.jpg.\n"
    "Read referral.txt, report the study the way findings-codebook.md defines it, and file the report:\n"
    "\n    python3 radreport.py report.json\n"
)

CLINICS = (
    "Westbridge Endodontic Referral Centre",
    "Harborview Dental Radiology",
    "Elm Street Oral Health Unit",
    "Crescent Park Dental Institute",
    "Northfield Community Dental Service",
    "Lakeside Endodontics",
)
DENTISTS = ("Abara", "Lindqvist", "Marchetti", "Nakamura", "Osei", "Pavlova", "Rao", "Schneider")

CODEBOOK = """# Findings codebook — periapical reporting

Report every periapical lesion. Box the lesion itself (the radiolucency at the root tip), not the
whole tooth, and grade it on the Periapical Index:

| Grade | Meaning |
| --- | --- |
| 3 | changes in bone structure with some mineral loss |
| 4 | periodontitis with well-defined radiolucent areas |
| 5 | severe periodontitis with exacerbating features |

Grades 1 (normal periapical structures) and 2 (small changes in bone structure) are not lesions
and are not reported. `pai-reference.jpg` shows example lesions of each reported grade from other
studies, each outlined the way a lesion is boxed.

Every study on this worklist has at least one periapical lesion.

## Report format

One JSON object: `{"lesions": [{"box": [x1, y1, x2, y2], "pai": 4}]}`.

- Box corners are fractions of the image width and height, measured from the top-left corner;
  keep every value between 0 and 1.
- One entry per lesion.

## Filing

Save the report as `report.json` and file it:

    python3 radreport.py report.json
"""

RADREPORT = '''"""Validate one periapical report (clinic reporting tool)."""

import hashlib
import json
import math
import sys


def number(value):
    """A finite JSON number, never a bool: NaN and Infinity parse, but no scorer can place them."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


path = sys.argv[1] if len(sys.argv) > 1 else "report.json"
try:
    report = json.load(open(path, encoding="utf-8"))
    lesions = report["lesions"]
    if not isinstance(lesions, list):
        raise ValueError("lesions is not a list")
    for lesion in lesions:
        box = lesion["box"]
        if len(box) != 4 or not all(number(v) for v in box):
            raise ValueError(f"bad box {box!r}")
        grade = lesion["pai"]
        if not number(grade) or not float(grade).is_integer():
            raise ValueError(f"bad grade {grade!r}")
except Exception as error:
    print(f"radreport: the report is not valid: {error}", file=sys.stderr)
    sys.exit(2)
digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:8]
print(f"Report D-{digest} validated.")
'''

PAI_REFERENCE = Path(__file__).with_name("pai-reference.jpg")
"""Example lesions of grades 3-5 from the train split; built by scripts/periapical-lesions/build_pai_reference.py."""

READ_TIMEOUT_S = 30

TIMEOUT = vf.TaskTimeout(setup=15 * 60, agent=2 * 60 * 60)
"""Applies only where the agent sets no timeout of its own (by default it sets none and waits forever): a hung setup or
agent then ends as an errored rollout instead of stalling the run."""


def _json_objects(text: str) -> list[str]:
    """Every balanced {...} span in `text`, in order."""
    spans, depth, start = [], 0, None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : index + 1])
                start = None
    return spans


def _is_number(value: object) -> bool:
    """A real JSON number, never a bool: the clinic validator's box rule."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_grade(value: object) -> bool:
    """A whole number, never a bool: the validator rejects 3.9, and the scorer must agree."""
    return _is_number(value) and (isinstance(value, int) or float(value).is_integer())


def parse_lesions(reply: str) -> list[dict]:
    """The lesion list of a filed report: the last readable lesion list wins (the last one inside a fenced code
    block, and without a readable fence the last one anywhere). Text without a readable lesion list counts as no
    report, and no report counts as finding nothing. Entries must carry the clinic validator's types — finite
    numeric boxes, whole-number grades (4.0 counts, 3.9 and booleans do not) — and invalid entries are dropped."""
    if not isinstance(reply, str):
        return []

    def read(candidate: str):
        try:
            lesions = json.loads(candidate)["lesions"]
            return [
                {"box": [float(v) for v in lesion["box"]], "pai": int(lesion["pai"])}
                for lesion in lesions
                if len(lesion["box"]) == 4 and all(_is_number(v) for v in lesion["box"]) and _is_grade(lesion["pai"])
            ]
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
            return None

    fenced = [span for block in re.findall(r"```(?:json)?(.*?)```", reply, re.DOTALL) for span in _json_objects(block)]
    readable = [span for span in fenced if read(span) is not None] or [
        span for span in _json_objects(reply) if read(span) is not None
    ]
    return read(readable[-1]) if readable else []


def box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2:] - boxes[:, :2]).clip(0).prod(-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union for every (a, b) pair, exact at the threshold; boxes without area share none."""
    overlap = (np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).clip(0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)


def match(lesions: list[dict], boxes: list[list[float]]) -> list[tuple[int, int]]:
    """One-to-one (predicted, reference) pairs that overlap by at least MATCH_IOU: the most pairs, then the tightest."""
    if not lesions or not boxes:
        return []
    iou = iou_matrix(np.array([lesion["box"] for lesion in lesions]), np.array(boxes))
    weight = np.where(iou >= MATCH_IOU, 1 + iou / iou.size, 0)  # a pair outweighs any IoU tie-break
    return [(p, r) for p, r in zip(*linear_sum_assignment(weight, maximize=True)) if iou[p, r] >= MATCH_IOU]


def film_scores(lesions: list[dict], boxes: list[list[float]], pai: list[int]) -> tuple[float, float]:
    """(lesion F1, graded-lesion F1). Both count extra boxes against the answer, so boxing everything scores ~0.
    A lesion-free film scores 1 for an empty report (the codebook's correct answer) and 0 for any finding."""
    pairs = match(lesions, boxes)
    total = len(lesions) + len(boxes)
    if total == 0:
        return 1.0, 1.0
    return 2 * len(pairs) / total, 2 * sum(lesions[p]["pai"] == pai[r] for p, r in pairs) / total


class PeriapicalData(vf.TaskData):
    boxes: list[list[float]]
    """Reference lesion boxes as [x1, y1, x2, y2] fractions of the image size."""

    pai: list[int]
    """Reference Periapical Index grade (3-5) for each box."""


class PeriapicalTask(vf.Task[PeriapicalData]):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        image_url = self.data.prompt[0].content[1].image_url.url
        await runtime.write("pantomograph.jpg", base64.b64decode(image_url.removeprefix(JPEG_DATA_URL)))
        await runtime.write("referral.txt", vignette(self.data.name).encode())
        await runtime.write("findings-codebook.md", CODEBOOK.encode())
        await runtime.write("pai-reference.jpg", PAI_REFERENCE.read_bytes())
        await runtime.write("radreport.py", RADREPORT.encode())

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """The filed report, read back through the runtime's file API after the agent stops — a
        guest-controlled `cat` must not decide what the clinic submitted. Anything that fails to
        read counts as no report."""
        try:
            submitted = await asyncio.wait_for(runtime.read("report.json"), READ_TIMEOUT_S)
        except (FileNotFoundError, asyncio.TimeoutError, vf.SandboxError):
            submitted = b""
        trace.info["submitted"] = submitted.decode(errors="replace")

    def scores(self, trace: vf.Trace) -> tuple[float, float]:
        return film_scores(parse_lesions(trace.info.get("submitted", "")), self.data.boxes, self.data.pai)

    @vf.reward(weight=0.5)
    async def lesion_f1(self, trace: vf.Trace) -> float:
        return self.scores(trace)[0]

    @vf.reward(weight=0.5)
    async def pai_f1(self, trace: vf.Trace) -> float:
        """F1 over lesions found and given the reference grade."""
        return self.scores(trace)[1]


class PeriapicalConfig(vf.TasksetConfig):
    split: str = "test"
    """Dataset split: train, validation or test (made per film; the source has no patient IDs)."""


class PeriapicalTaskset(vf.Taskset[PeriapicalTask, PeriapicalConfig]):
    def load(self) -> Iterator[PeriapicalTask]:
        split = self.config.split
        rows = load_dataset(DATASET, data_files={split: f"data/{split}-*.parquet"}, split=split, revision=REVISION)
        for idx, row in enumerate(rows.cast_column("image", Image(decode=False))):
            yield PeriapicalTask(to_data(idx, row), self.config.task)


def vignette(image_id: str) -> str:
    """The referral for one study, fixed by the film so every rerun of a task sees the same paperwork.

    The source has no patient data, so the note carries only administrative fields that cannot contradict the
    film or hint at its findings: no age, sex, symptoms or history, and one reason for every study."""
    seed = hashlib.sha256(image_id.encode()).digest()
    return (
        f"{CLINICS[seed[0] % len(CLINICS)]}\n"
        f"Received 2026-{1 + seed[5] % 12:02d}-{1 + seed[6] % 28:02d}\n"
        "\n"
        f"Patient P-{seed[:2].hex().upper()}\n"
        "Study: dental pantomograph (one exposure)\n"
        f"Referring clinician: Dr {DENTISTS[seed[1] % len(DENTISTS)]}\n"
        "Reason: radiographic review\n"
    )


def to_data(idx: int, row: dict) -> PeriapicalData:
    w, h = row["width"], row["height"]
    image = base64.b64encode(row["image"]["bytes"]).decode()
    content = [
        {"type": "text", "text": WORKLIST_PROMPT},
        {"type": "image_url", "image_url": {"url": JPEG_DATA_URL + image}},
    ]
    return PeriapicalData(
        idx=idx,
        name=row["image_id"],
        prompt=[{"role": "user", "content": content}],
        image=SANDBOX_IMAGE,
        network_allow=[],
        timeout=TIMEOUT,
        boxes=[[x / w, y / h, (x + bw) / w, (y + bh) / h] for x, y, bw, bh in row["objects"]["bbox"]],
        pai=[label + 3 for label in row["objects"]["pai"]],
    )
