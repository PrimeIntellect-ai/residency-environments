# Synthetic Workspace Gym generators

This directory contains the deterministic generation, provenance, and validation library for the Synthetic Workspace Gym taskset. It is deliberately separate from the installable environment package.

The checked-in source curricula live in `configs/synthetic_workspace_gym/`. Earlier revisions remain frozen for provenance. The six V5 curricula and evaluation panels calibrate untouched baselines through the shipped execution protocol and harden structure-only and nondeterministic rewards:

- `sft-easy-v5`
- `sft-validation-v5`
- `rl-hard-v5`
- `rl-eval-v5`
- `eval-d1-d4-paired-panel-48-v5`
- `eval-d5-family-calibration-40-v5`

Standalone generation and calibration entrypoints live in `scripts/synthetic_workspace_gym/`. From the repository root, rebuild the curricula with:

```bash
uv run --project generators/synthetic_workspace_gym python scripts/synthetic_workspace_gym/build_qwen35_training_manifests.py
```

Generate the review-safe artifact layout from all 1,328 assignments, validating each reference solution while exporting:

```bash
uv run --project generators/synthetic_workspace_gym swg-export-dataset \
  --manifest-dir configs/synthetic_workspace_gym \
  --output-dir generators/synthetic_workspace_gym/dist \
  --generator-revision "$(git rev-parse HEAD)"
```

`--generator-revision` must be the full committed source revision used for the export. It is embedded in the export metadata and every task manifest. Generation writes canonical UTF-8/LF bytes so the same revision produces identical task fingerprints on Windows and Linux.

The export deliberately separates trust domains. `dist/dataset/` contains lightweight curriculum indexes and one deterministic grader-only blob per unique task. `dist/images/<curriculum>/` contains only the visible workspaces for that curriculum. `integrity.json` records every exported file digest. Hidden files and visible workspaces are never combined in one distributable archive.

Publish the dataset directory privately and retain the immutable commit returned by Hugging Face. Because the dataset contains grader-only payloads, public visibility requires the explicit `--public` flag:

```bash
uv run --project generators/synthetic_workspace_gym swg-publish-dataset \
  --dataset-dir generators/synthetic_workspace_gym/dist/dataset \
  --repo-id <organization>/<dataset>
```

Build the six split-isolated agent images and the dedicated grader image with Prime's server-side builder. The Dockerfiles pin their Python base image by digest, and the generated map resolves every pushed image to an immutable digest:

```bash
uv run --project generators/synthetic_workspace_gym \
  python scripts/synthetic_workspace_gym/build_images.py \
  --artifacts generators/synthetic_workspace_gym/dist \
  --tag <artifact-version> \
  --output generators/synthetic_workspace_gym/dist/task_images.json
```

Run static checks and build the generator package with:

```bash
uv run ruff check generators/synthetic_workspace_gym scripts/synthetic_workspace_gym
uv run ruff format --check generators/synthetic_workspace_gym scripts/synthetic_workspace_gym
uv build generators/synthetic_workspace_gym
```
