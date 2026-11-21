"""Build the Periapical Index reference chart shipped with the taskset.

The index is defined by reference radiographs, so the chart shows example lesions of each reported grade
(3, 4, 5) from the train split. Examples are the first lesions of a fixed hash ranking, one per film, with
no selection by appearance. Each tile is a square region around one lesion, with the lesion's reference box
outlined and captioned with its size on the original film. Re-run to reproduce `periapical_lesions/pai-reference.jpg` byte for byte.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from periapical_lesions.taskset import DATASET, REVISION
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "environments/periapical-lesions/periapical_lesions/pai-reference.jpg"
PER_GRADE = 4
TILE = 300
CONTEXT = 3.0
MIN_SIDE = 200
GAP = 12
LABEL_W = 150
TITLE_H = 44


def rank(image_id: str, index: int) -> str:
    return hashlib.sha256(f"pai-reference\0{image_id}\0{index}".encode()).hexdigest()


def main() -> None:
    rows = load_dataset(DATASET, data_files={"train": "data/train-*.parquet"}, split="train", revision=REVISION)
    rows = rows.cast_column("image", HFImage(decode=False))
    candidates = {3: [], 4: [], 5: []}
    for position, objects in enumerate(rows["objects"]):
        for index, label in enumerate(objects["pai"]):
            candidates[label + 3].append((position, index))
    ids = rows["image_id"]
    chosen = {}
    for grade, items in candidates.items():
        items.sort(key=lambda item: rank(ids[item[0]], item[1]))
        picked, films = [], set()
        for position, index in items:
            if position not in films:
                films.add(position)
                picked.append((position, index))
            if len(picked) == PER_GRADE:
                break
        chosen[grade] = picked
    width = LABEL_W + PER_GRADE * (TILE + GAP) + GAP
    height = TITLE_H + 3 * (TILE + GAP) + GAP
    chart = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(chart)
    draw.text(
        (GAP, 10),
        "Periapical Index reference: example lesions from other studies, each outlined (magnifications differ)",
        fill=255,
        font=ImageFont.load_default(size=22),
    )
    provenance = {}
    for row_index, grade in enumerate((3, 4, 5)):
        top = TITLE_H + GAP + row_index * (TILE + GAP)
        draw.text((GAP, top + TILE // 2 - 20), f"Grade {grade}", fill=255, font=ImageFont.load_default(size=30))
        provenance[grade] = []
        for column, (position, index) in enumerate(chosen[grade]):
            row = rows[position]
            film = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            x, y, bw, bh = row["objects"]["bbox"][index]
            side = min(max(CONTEXT * max(bw, bh), MIN_SIDE), film.width, film.height)
            left = min(max(x + bw / 2 - side / 2, 0), film.width - side)
            upper = min(max(y + bh / 2 - side / 2, 0), film.height - side)
            tile = film.crop((round(left), round(upper), round(left + side), round(upper + side))).resize(
                (TILE, TILE), Image.LANCZOS
            )
            scale = TILE / side
            box = [(x - left) * scale, (y - upper) * scale, (x + bw - left) * scale, (y + bh - upper) * scale]
            ImageDraw.Draw(tile).rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], outline=255, width=2)
            caption = f"{round(bw)} x {round(bh)} px on the film"
            pen = ImageDraw.Draw(tile)
            pen.rectangle([0, 0, 8 + pen.textlength(caption, font=ImageFont.load_default(size=16)), 24], fill=0)
            pen.text((4, 3), caption, fill=255, font=ImageFont.load_default(size=16))
            chart.paste(tile, (LABEL_W + GAP + column * (TILE + GAP), top))
            provenance[grade].append({"image_id": row["image_id"], "lesion_index": index})
    chart.save(OUT, format="JPEG", quality=92)
    print(json.dumps({"path": str(OUT), "bytes": OUT.stat().st_size, "size": chart.size, "examples": provenance}))


if __name__ == "__main__":
    main()
