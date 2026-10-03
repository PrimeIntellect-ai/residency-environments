# Synthetic Workspace Gym V1

Synthetic Workspace Gym (SWG) is a native Verifiers V1 taskset for training terminal agents on generated, executable workspaces. Tasks cover tabular transformations, script repair, data pipelines, local-document retrieval, and composite workflows. Each task has a deterministic assignment, an agent-visible workspace, and a trusted evaluator with partial-credit scoring.

## Design

`SyntheticWorkspaceTaskset` loads prompts and assignments from a Hugging Face dataset pinned to an immutable commit. Each assignment becomes a typed `SyntheticWorkspaceTask` with typed `SyntheticWorkspaceData` and `SyntheticWorkspaceState`. Generation and calibration code lives separately under `generators/synthetic_workspace_gym`; it is not installed with this environment.

The task declares `NEEDS_CONTAINER = True` and a working directory of `/workspace`. Verifiers and the selected standard harness own model interaction, terminal access, runtime provisioning, rollout timeouts, and trace capture. SWG does not wrap a harness or implement a rollout loop; it adds a shorter timeout only around candidate programs executed during trusted finalization.

Visible workspaces are baked into six immutable Prime VM artifacts, one per curriculum or evaluation panel. Each pin records the public runtime tag together with Prime's artifact ID and CAS path. Setup selects one workspace and removes the rest of the split payload before the agent starts. Hidden assets never enter the agent runtime or task state.

Finalization uses three trust domains:

1. The completed workspace is inventoried by an absolute trusted interpreter and read through the runtime file API under file-count, directory-count, per-file, and aggregate-byte limits. Symlinks and non-regular entries are rejected.
2. A fresh runtime from the same split image executes candidate code with test inputs but no expected answers. Candidate commands run in a timeout-managed process group; timeout sends `TERM`, escalates to `KILL`, records return code `124`, skips an unavailable determinism rerun, and continues to grading. Concrete observations and profile outputs travel in a host-controlled payload outside `/workspace`; stdout never carries score-bearing data.
3. A second fresh runtime from the dedicated grader image receives those artifacts and the grader-only dataset payload. It computes reward without executing agent-controlled code.

Both temporary runtimes use `vf.provision_runtime`, are network-restricted, and are torn down automatically. The weighted `workspace_score` reward is the evaluator's normalized score in `[0, 1]`; metrics report binary success, changed/final file counts, and evaluator subscores.

The task requests framework-only network access. A fresh runtime is used for every rollout, and generated paths are validated before being written or extracted.

## Repository layout

The paired generator contribution keeps reusable generation and validation libraries in `generators/synthetic_workspace_gym/`, image definitions and build helpers in `scripts/synthetic_workspace_gym/`, and source curriculum configurations in `configs/synthetic_workspace_gym/`. Generated manifests, hidden trees, and visible workspaces are not committed to `environments/`.

## Install

From the repository root:

```bash
uv pip install -e environments/synthetic_workspace_gym
```

## Smoke evaluation

Run the required small integration smoke with the standard harness/runtime:

```bash
uv run eval --taskset.id synthetic-workspace-gym -n 1 -r 2 --max-turns 4
```

Use the default `sft-easy-v5` mid-training manifest or select another immutable dataset manifest and filters:

```bash
uv run eval --taskset.id synthetic-workspace-gym \
  --env.taskset.manifest rl-eval-v3 \
  --env.taskset.families '["script_repair","pipeline"]' \
  --env.taskset.difficulties '[4,5]' \
  -n 20 -r 3 --rich false --no-push
```

The purpose-specific manifests are `sft-easy-v5` (family-calibrated D1-D3 teacher traces), `sft-validation-v5` (scenario- and seed-held-out D1-D3), `rl-hard-v5` (family-calibrated D4-D5 atomic and composite training), and `rl-eval-v5` (family-stratified held-out D4-D5 scenarios, seeds, fixtures, and compositions). V5 records the exact public generator commit, calibrates untouched rewards through the shipped execution protocol, requires deterministic command outputs for reward, and transports observations through a trusted post-process frame. Verifiers' normal `-n`, `-r`, `--shuffle`, harness, model, sampling, and runtime options apply without environment-specific rollout arguments.

For difficulty calibration, use `eval-d1-d4-paired-panel-48-v5` and `eval-d5-family-calibration-40-v5`. The first holds family, scenario, and seed fixed while difficulty changes; the second reports D5 separately by family and retrieval profile.

## Configuration

| Field                                      | Default       | Purpose                                                        |
| ------------------------------------------ | ------------- | -------------------------------------------------------------- |
| `manifest`                                 | `sft-easy-v5` | Immutable curriculum or evaluation panel                       |
| `families`                                 | all           | Optional family filter                                         |
| `difficulties`                             | all           | Optional difficulty filter (`1` through `5`)                   |
| `tasks`                                    | all           | Optional exact task-ID filter                                  |
| `task.max_workspace_files`                 | `1024`        | Maximum number of regular files transferred                    |
| `task.max_workspace_directories`           | `1024`        | Maximum number of workspace directories                        |
| `task.max_file_bytes`                      | `8388608`     | Maximum workspace file or execution artifact transfer          |
| `task.max_workspace_bytes`                 | `67108864`    | Maximum aggregate workspace transfer                           |
| `task.max_result_bytes`                    | `2097152`     | Maximum structured grading result returned from the runtime    |
| `task.candidate_execution_timeout_seconds` | `120`         | Per-run candidate execution limit inside finalization           |
| `task.candidate_kill_grace_seconds`        | `5`           | Grace period between candidate process-group termination signals |

The package pins the dataset commit and all seven runtime images in `synthetic_workspace_gym/artifacts.json`. Changing task data, task behavior, or reward semantics requires publishing new immutable artifacts and bumping the package version.
