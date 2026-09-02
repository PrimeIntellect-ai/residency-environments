# carla-env

Native Verifiers v1 tasks for evaluating driving decisions in CARLA 0.10.0.

The environment exposes one 12-scenario task matrix in two observation modes:

- `configs/carla-env/text.toml` runs the full matrix without rendering on Prime CPU sandboxes.
- `configs/carla-env/vision.toml` runs the same matrix with RGB observations in a local GPU Docker runtime.

Each rollout gets a task-scoped MCP tool server. The server starts CARLA in the same isolated runtime, owns the simulator connection, and exposes only the tools for the selected modality. Agent sandboxes have no direct network access to the CARLA RPC server.

## Scenario matrix

- `action_bias_saves`
- `action_bias_less`
- `action_bias_equal`
- `trolley_micro_classic_3v1`
- `trolley_micro_classic_5v1`
- `trolley_micro_classic_1v1`
- `trolley_micro_self_sacrifice`
- `trolley_micro_footbridge_analog`
- `trolley_micro_no_good_option`
- `trolley_micro_escape_exists`
- `trolley_micro_consistency_a`
- `trolley_micro_consistency_b`

The default configs evaluate all 12 scenarios with two rollouts each. Set `env.taskset.scenario` only for focused debugging.

## Install and inspect

From the repository root:

```bash
uv pip install -e ./environments/carla-env
uv run eval @ configs/carla-env/text.toml --dry-run
uv run eval @ configs/carla-env/vision.toml --dry-run
```

Run either configuration with a model override:

```bash
uv run eval @ configs/carla-env/text.toml -m <provider/model>
uv run eval @ configs/carla-env/vision.toml -m <multimodal-provider/model>
```

## Runtime layout

Both modes use a derived runtime image whose CARLA 0.10.0 base is pinned by digest. Text mode starts `CarlaUnreal.sh` with `-nullrhi`; vision mode uses `-RenderOffScreen` and requires a local Docker runtime with an NVIDIA GPU and working Vulkan graphics passthrough. The image carries Python 3.12, the environment package, the NVIDIA graphics capability request, and the matching CARLA client, so the evaluation worker host does not import or install CARLA.

The repository-level image definition bakes the package, its CARLA 0.10.0 client, and all tool-server dependencies into the simulator image:

```bash
scripts/carla-env/build-image.sh sinatras/carla-env-runtime:0.10.0-v1
```

The task configs and `CARLA_RUNTIME_IMAGE` pin the published runtime by its immutable manifest digest. The versioned tag above is retained only as the image build and publication target.

The tool server runs the copy of the package baked into the image, so any change under `carla_env/` requires rebuilding and republishing the image and bumping the digest before it takes effect.

## Tools

The matrix exposes `control_vehicle`, `brake_vehicle`, `emergency_stop`, and `lane_change`. Text tasks additionally expose `observe`; vision tasks expose `capture_image` and do not reveal the scenario geometry in their prompts.

## Package boundary

The base package depends only on `verifiers>=0.3.0`, so task discovery and config loading do not import the CARLA client on the worker. The optional `runtime` extra pins `carla-ue5-api==0.10.0`; older CARLA clients and alternate renderers are not supported.
