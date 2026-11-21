# periapical-lesions

A Verifiers v1 taskset for locating periapical lesions in panoramic dental radiographs and assigning each a Periapical Index grade (PAI 3, 4 or 5). Each film contains at least one annotated lesion.

The agent sees the radiograph in the prompt and gets the same JPEG, an administrative referral, a grading codebook, and a reference chart in a network-restricted sandbox. It writes `report.json` with one normalized `[x1, y1, x2, y2]` box and integer `pai` per lesion. `python3 radreport.py report.json` validates the file; the task scores its final contents, not the chat reply. A missing or unreadable report counts as no findings.

## Scoring

Predictions are matched one-to-one to reference boxes at IoU ≥ 0.30, maximizing the number of matches before overlap. Grades are not used for matching. The reward is the mean of detection F1 and grade-matched F1. Both use `predicted + reference` as the denominator, so extra boxes reduce both scores. A perfect report must find every lesion, assign every grade correctly and contain no extras.

## Run

From the repository root, with Prime sandbox access and an image-capable model endpoint:

```bash
uv sync --locked
uv pip install -e ./environments/periapical-lesions
uv run vf-eval @ configs/periapical-lesions/pi.toml -m <model> -n 1 -r 2
```

Pi is the default harness. The [Pi config](../../configs/periapical-lesions/pi.toml) pins Pi 0.84.1; a [Claude Code config](../../configs/periapical-lesions/eval.toml) is also available. The default split is `test`; set `--env.taskset.split train` or `validation` to change it. The declared full evaluation is 387 films, one rollout per film. The sandbox image is `prime/dylantirandaz/periapical-lesions:v1` ([Dockerfile](../../scripts/periapical-lesions/Dockerfile)).

## Data and license

The [pinned dataset](https://huggingface.co/datasets/tirandazdylan/periapical-lesions-pai/tree/3c74b963b449ae66ef23266a38812688dbe8cc81) contains 3,087 training, 389 validation, and 387 test films. Splits are by film; the source does not supply patient identifiers. The host loads the dataset; the full films and scoring annotations are not bundled in the package. The bundled grading chart contains four training-set crops per grade ([build script](../../scripts/periapical-lesions/build_pai_reference.py)).

Code: [Apache-2.0](licenses/LICENSE). Radiographs, annotations and derived chart: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from [Do et al.](https://doi.org/10.17632/kx52tk2ddj.3); see [NOTICE](licenses/NOTICE) for attribution and adaptations. This is a research benchmark, not a clinical diagnostic tool.
