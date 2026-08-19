# carla-env

CARLA verifiers environment for autonomous-driving evaluation and RL.

| Field | Value |
| --- | --- |
| **Environment ID** | `carla-env` |
| **Version** | `0.3.0` |
| **Type** | Verifiers v1 taskset backed by `StatefulToolEnv` |

## Overview

Scenario families:

- `trolley_micro_*`: benchmark trolley dilemmas
- `action_bias_*`, `bias_*`: action-vs-inaction trolley variants
- `maze`: hidden-goal text navigation
- `navigation*`: text-first open navigation with optional NPC vehicles/pedestrians
- `navigation_vision*`: vision-first navigation with RGB camera access
- `free_roam*`: open-ended driving with optional traffic

Renderer modes:

- Native CARLA RGB cameras
- NuRec neural rendering on CARLA 0.9.16
- Cosmos Transfer2.5 stylized RGB via remote frame server

Observation modes:

- Text-first: trolley, action-bias, maze, `navigation*`
- Vision-first: `navigation_vision*`

## Quickstart

```bash
# Recommended today: run the validated Prime H100 flow.
environments/carla-env/scripts/e2e_prime_h100_runtime.sh --stack both --runtime-image sinatras/carla-env-runtime:latest

# Or run only NuRec / only Cosmos
environments/carla-env/scripts/e2e_prime_h100_runtime.sh --stack nurec --runtime-image sinatras/carla-env-runtime:latest
environments/carla-env/scripts/e2e_prime_h100_runtime.sh --stack cosmos --runtime-image sinatras/carla-env-runtime:latest
```

## Local Install

The old “install one extra and immediately run” path is no longer the right
default. The validated package install today is:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel

# Validated local client path
uv pip install -e "./environments/carla-env[carla9]"

# If you need NuRec support
uv pip install -e "./environments/carla-env[carla9,nurec]" "imageio[ffmpeg]" "huggingface_hub"

# CARLA 0.10.0-only workflows still exist, but are not the main validated path
# uv pip install -e "./environments/carla-env[carla10]"
```

For Hugging Face-backed features like NuRec scene download and Cosmos gated
weights, keep a repo-local `.env` with a valid token:

```bash
cat > .env <<'EOF'
HF_TOKEN=hf_your_token_here
EOF
```

## Local CARLA Server

Start a local CARLA 0.9.16 server explicitly before running the env:

```bash
docker run --rm \
  --gpus all \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --shm-size=8g \
  -e XDG_RUNTIME_DIR=/tmp/runtime-carla \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  /bin/sh -lc 'mkdir -p /tmp/runtime-carla && chmod 700 /tmp/runtime-carla && cd /workspace && exec ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -stdout -FullStdOutLogOutput -unattended'
```

Then run examples:

```bash
uv run eval carla-env -m "openai/gpt-4.1-mini" \
  --env.taskset.env-args '{"scenario": "maze", "sandbox": {"mode": "disabled"}, "host": "127.0.0.1", "port": 2000, "carla_version": "0.9.16"}' \
  -n 1 -r 1

uv run eval carla-env -m "openai/gpt-4.1-mini" \
  --env.taskset.env-args '{"scenario": "navigation_Town10HD_v1_p1", "sandbox": {"mode": "disabled"}, "host": "127.0.0.1", "port": 2000, "carla_version": "0.9.16"}' \
  -n 1 -r 1

uv run eval carla-env -m "qwen/qwen3-vl-8b-instruct" \
  --env.taskset.env-args '{"scenario": "navigation_vision_Town10HD_v1_p1", "sandbox": {"mode": "disabled"}, "host": "127.0.0.1", "port": 2000, "carla_version": "0.9.16"}' \
  -n 1 -r 1
```

## Pod Local Server

On some GPU pods, nested Docker cannot unpack or launch the official `carlasim/carla`
images even though the node has a usable NVIDIA GPU. For that case, use
[`scripts/start_local_carla_on_pod.sh`](scripts/start_local_carla_on_pod.sh), which:

- installs the required host packages
- reuses a preinstalled CARLA root when available, or unpacks the official CARLA image onto the host
- launches CARLA directly from the extracted rootfs on the host as a non-root user
- uses `/ephemeral` or `/workspace` for large CARLA cache/rootfs data
- can apply a Vulkan ICD override when the provider image needs it

Example:

```bash
sudo environments/carla-env/scripts/start_local_carla_on_pod.sh start --mode vision --port 4000

uv run eval carla-env -m "qwen/qwen3-vl-8b-instruct" \
  --env.taskset.env-args '{"scenario":"navigation_vision_Town10HD_v1_p1","sandbox":{"mode":"disabled"},"host":"127.0.0.1","port":4000,"carla_version":"0.10.0"}' \
  -n 1 -r 1
```

## Stage 3 Rendering

NuRec:

- enable with `enable_nurec=True` and `nurec_scene_path=...`
- forces `carla_version="0.9.16"`
- supports `nurec_mode="replay"` and `nurec_mode="drive"`

Cosmos:

- enable with `enable_cosmos=True` and `cosmos_server_url="http://host:8080"`
- keeps native CARLA depth capture when the depth sensor is available
- degrades to raw CARLA RGB if the Cosmos server is unhealthy or stylization fails

Operational helper:

- [`scripts/cosmos_frame_server.py`](scripts/cosmos_frame_server.py) runs a long-lived FastAPI frame server for Cosmos stylization on a GPU host

## Environment Args

| Argument | Default | Description |
| --- | --- | --- |
| `scenario` | `"action_bias_saves"` | Scenario identifier |
| `host` | `$CARLA_HOST` or `127.0.0.1` | CARLA host |
| `port` | `$CARLA_PORT` or `2000` | CARLA port |
| `dataset_path` | `None` | Custom JSONL dataset |
| `trolley_micro_scoring` | `"expected"` | `"expected"` or `"actual"` |
| `sandbox` | `{"mode":"prime"}` | Prime sandbox config or local mode |
| `traffic_manager_enabled` | `None` | Explicit TrafficManager opt-in/out |
| `carla_version` | inherited | `0.10.0`, `0.9.16`, or `auto` (local only) |
| `enable_nurec` | `False` | Enable NuRec neural rendering |
| `nurec_scene_path` | `None` | Path to NuRec USDZ scene |
| `nurec_mode` | `None` | `replay` or `drive` |
| `enable_cosmos` | `False` | Enable Cosmos stylization |
| `cosmos_server_url` | `None` | Cosmos frame server URL |

## Scenario Names

| Scenario | Description |
| --- | --- |
| `maze` | Hidden-goal text navigation |
| `navigation` | Text-first open navigation |
| `navigation_<Map>_v<N>_p<M>` | Text-first navigation with map, vehicles, pedestrians |
| `navigation_vision` | Vision-first open navigation |
| `navigation_vision_<Map>_v<N>_p<M>` | Vision-first navigation with map, vehicles, pedestrians |
| `action_bias_saves` | Swerve avoids all pedestrians |
| `action_bias_less` | Swerve hits fewer |
| `action_bias_equal` | Equal harm either way |
| `bias_<C>v<S>` | Custom action-bias setup |
| `trolley_micro_<id>` | Benchmark trolley scenario |

## Tools

Common tools:

- `control_vehicle(throttle, steer)`
- `brake_vehicle(intensity)`
- `emergency_stop()`
- `lane_change(direction)`
- `init_navigation_agent(behavior)`
- `set_destination(x, y, z)`
- `follow_route(steps)`
- `get_goal_info()`

Text-first only:

- `observe()`

Vision-first only:

- `capture_image()`

Cosmos-enabled vision episodes only:

- `capture_depth()`

`get_goal_info()` behavior:

- `maze` and `navigation*`: `distance_to_goal_m=... direction=... improving=...`
- `navigation_vision*`: `distance_to_goal_m=... improving=...`

## Notes

- The validated install path today is Python 3.12 + `.[carla9]`.
- `.[carla10]` is still available for CARLA 0.10.0-only workflows, but it is not the main validated renderer path.
- NuRec requires `.[carla9,nurec]`, a valid `nurec_scene_path`, and Hugging Face auth.
- The most reliable full deployment path today is the Prime H100 wrapper in `scripts/e2e_prime_h100_runtime.sh`.
- `navigation*` defaults to `traffic_manager_enabled=False`. TrafficManager is opt-in only.
- Sandbox startup defaults are mode-aware:
  - text-only scenarios: `./CarlaUnreal.sh -nullrhi -nosound`
  - vision scenarios: `./CarlaUnreal.sh -RenderOffScreen -nosound`
- `navigation_vision*` does not expose normal text observations.
- `navigation_vision*` does not include destination coordinates in the prompt.
- `import carla_env` works without CARLA installed, but loading the environment still requires one of the CARLA client extras.
