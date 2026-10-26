# MIDOG Mitosis

An agent looks at a microscope image of tumor tissue and must find every mitotic figure: a cell caught in the middle of dividing. Pathologists count these cells to judge how fast a tumor grows. The images and expert labels come from [MIDOG++](https://doi.org/10.1038/s41597-023-02327-4).

Each task is one 768 × 768 px field of a tissue slide stained with H&E (the standard stain), at 0.25 µm per pixel, or two fields of one slide side by side. The agent ends its reply with `Answer: [[x, y], ...]`, one point per figure, with x and y from 0 to 1000 across the image's width and height. The reward is 1 if the points match the expert-labeled figures one-to-one, each within 30 px (7.5 µm), with none missing and none extra; otherwise 0. The agent is not told how many figures there are.

## Data

[`tirandazdylan/midog-mitosis`](https://huggingface.co/datasets/tirandazdylan/midog-mitosis) (CC0), pinned to commit [`d9616cb2`](https://huggingface.co/datasets/tirandazdylan/midog-mitosis/tree/d9616cb2d5207d9b06943c0a263dbd4025228bea) by `DATASET_REVISION` in [`taskset.py`](midog_mitosis/taskset.py). Fields are used only where the first two MIDOG++ experts agreed on every labeled cell.

| Config   | What it holds                                     | Train | Validation |
| -------- | ------------------------------------------------- | ----: | ---------: |
| `tasks`  | Images with three or more figures                 |   357 |         64 |
| `warmup` | Images with one figure, for training small models |   543 |         37 |

The benchmark is 100 test tasks (`-n 100`) from all seven tumor types.

Splits are by whole slide, so no slide appears in two splits. Test holds all 111 official MIDOG++ test slides plus slides held out from the official training set (a custom split, not the official one); the 100 benchmark tasks, 97 single fields and 3 pairs, come from 55 of them. Fields never overlap, no figure appears in two `tasks` tasks, and no warmup image shows a `tasks` field. Fields are chosen from the labels with fixed random seeds, never from model results. How the dataset is built and checked: [`generators/midog-mitosis`](../../generators/midog-mitosis).

## Run

The environment gives the agent no tools. A coding harness, an agent program such as Pi, Claude Code or Codex, plays each task with its own tools. Every attempt runs in a fresh Prime Intellect sandbox, an isolated container holding only the prompt and `image.jpg` (with Python, Pillow and NumPy). The answers never enter it, and it cannot reach the internet except the model's API (DNS lookups still resolve). You need a model that reads images, `verifiers>=0.3.1` (the framework this environment runs on; its CLI is `eval`, `vf-eval` in newer versions) and a `PRIME_API_KEY`.

```bash
uv pip install -e ./environments/midog-mitosis
# quick check: 1 task, 2 attempts
uv run --no-sync eval @ configs/midog-mitosis/pi.toml -m anthropic/claude-opus-5.5 --sampling.reasoning-effort high -n 1 -r 2
# full evaluation: start the image relay (see Results), then send the model's requests through it
uv run scripts/midog-mitosis/jpeg_proxy.py &
uv run --no-sync eval @ configs/midog-mitosis/pi.toml -m anthropic/claude-opus-5.5 --sampling.reasoning-effort high -n 100 -r 5 \
  --client.base-url http://127.0.0.1:8899/api/v1
uv run scripts/midog-mitosis/report.py outputs/<run-dir>          # pass rates with intervals
uv run --no-sync validate --taskset.id midog-mitosis --only-gold  # every expected answer scores 1
```

`-n 100 -r 5` runs the benchmark, five attempts per task, as reported below. [`pi.toml`](../../configs/midog-mitosis/pi.toml) gives each attempt an 8 GB sandbox and reruns it, up to twice, if the infrastructure fails; [`claude_code.toml`](../../configs/midog-mitosis/claude_code.toml) and [`codex.toml`](../../configs/midog-mitosis/codex.toml) run Claude and OpenAI models in their own harnesses. Recorded but not rewarded: `find/exact`, `find/f1`, `find/precision`, `find/recall`, `find/answered` (the reply parsed) and `find/predicted` (points listed). Taskset options (`--env.taskset.*`): `dataset`, `revision`, `config` (`tasks` or `warmup`), `split`. Don't evaluate a model trained on this release's test slides.

## Results

Every model ran in Pi at `high` reasoning effort on the 100 benchmark tasks, five attempts each, with requests sent through the image relay (October 8–9, 2026; Gemini 4 Argon and GPT-6 Astra on October 19–20). All 500 attempts per model were scored.

| Model               | pass@1 | pass@5 |
| ------------------- | ------ | ------ |
| Gemini 4 Argon      | 32.4%  | 43.0%  |
| Claude Opus 5.5     | 28.6%  | 38.0%  |
| GPT-6 Astra         | 22.8%  | 35.0%  |
| GLM 5.3 Flash       | 16.8%  | 41.0%  |
| DeepSeek V4.1 Flash | 8.8%   | 26.0%  |

pass@1 is the chance that one attempt solves a task; pass@5, that at least one of five does. These 100 tasks are the whole benchmark, so the only noise is each model's randomness from one attempt to the next: about ±2 to ±3 points for pass@1 and ±2 to ±5 for pass@5 (95%). Gemini 4 Argon's 3.8-point pass@1 lead over Opus is outside that noise (paired by task: +0.9 to +6.7), and so is Opus's 5.8-point lead over GPT-6 Astra (+3.0 to +8.6); the pass@5 gaps between Argon, GLM and Opus are within it. An answer that ignores the image solves no task.

**Relay:** [`jpeg_proxy.py`](../../scripts/midog-mitosis/jpeg_proxy.py), a small local server between the eval and Prime's model API, re-encodes every image that isn't already JPEG as JPEG (quality 95). It is needed because Pi resends all images the agent has opened on every turn, and Prime can reject requests with more than about 24 MB of images; without the relay, half of DeepSeek's attempts failed on that limit. Opus, which never reached the limit, scored the same within noise without it. The relay also retries a request that Prime fails with a timeout inside its response stream, which Pi would otherwise take for an empty reply (4.5% of Gemini 4 Argon's requests, 1 of GPT-6 Astra's 9,493).

Protocol, failure analysis, costs, audits and dataset checks: [`RESULTS.md`](../../scripts/midog-mitosis/RESULTS.md).

## Limitations

- The answers are the MIDOG++ expert labels; a dividing cell the experts missed counts as an extra point.
- A research benchmark, not a clinical tool.

## Citation

Code [Apache-2.0](LICENSE); data CC0, derived from MIDOG++. Cite Aubreville et al., *A comprehensive multi-domain dataset for mitotic figure detection*, Scientific Data 10, 484 (2023), [10.1038/s41597-023-02327-4](https://doi.org/10.1038/s41597-023-02327-4). The 30 px matching radius follows the MIDOG challenge ([10.5281/zenodo.15077361](https://doi.org/10.5281/zenodo.15077361)).
