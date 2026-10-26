# AerialSim

**A simulation environment for training and evaluating vision-language agents on autonomous drone missions.**

AerialSim is built on [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) and Unreal Engine 5.5. It provides a photorealistic urban park where agents use visual observations, spatial reasoning, planning, and flight controls to complete drone missions.

Agents navigate toward landing pads, identify obstacles, inspect locations, and respond to changes in the environment. Missions are evaluated using recorded flight trajectories and verifiable outcomes.

AerialSim supports two primary control modes: **AerialSim-Coding** and **AerialSim-Visual**. Both share the same simulator, scenario families, and trajectory-based evaluation framework.

## Environment modes

### AerialSim-Coding

In AerialSim-Coding, the agent writes a Python program to control the drone. It can execute practice flights, inspect simulator feedback, and revise its program before submitting a final solution.

The final program is executed in a fresh simulator instance and evaluated against the mission requirements.

This mode supports code generation, visual grounding, planning, and learning from simulator feedback.

### AerialSim-Visual

In AerialSim-Visual, the agent controls the drone one action at a time.

At each step, the agent receives a camera frame and selects its next flight action. The simulator executes the action and returns a new observation.

This mode supports closed-loop control, where the agent continuously updates its decisions based on visual feedback.

### Available environments

| Environment ID     | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `aerialsim-coding` | Python flight programs with simulator feedback and revision               |
| `aerialsim-visual` | Interactive flight control using sequential visual observations           |
| `aerialsim-flight` | Original environment for basic navigation, hovering, and landing missions |

The original `aerialsim-flight` environment uses Python programs to complete predefined flight objectives. It remains available alongside the two visual scenario environments.

## Visual scenario families

AerialSim includes seven scenario families that evaluate different aspects of visual perception, spatial reasoning, planning, and control.

| Scenario            | Objective                                                                 | Capability evaluated                     |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------------------- |
| `land_or_hold`      | Approach a landing pad and decide whether landing is safe.                | Visual decision-making                   |
| `find_pad`          | Locate a landing pad without being given its coordinates.                 | Visual grounding and exploration         |
| `obstacle_bypass`   | Identify and navigate through an unobstructed corridor.                   | Obstacle perception and route planning   |
| `alternate_pad`     | Choose a usable landing pad, or abort when neither is usable.             | Goal selection and replanning            |
| `patrol_inspection` | Carry a visual cue seen early in the route to a decision made at the end. | Visual memory and sequential planning    |
| `go_around`         | Respond to a timed route closure during flight.                           | Adaptive control and replanning          |
| `compound`          | Complete missions involving multiple conditions and possible outcomes.    | Multi-step reasoning and decision-making |

## Flight evaluation

AerialSim records the drone's trajectory throughout each mission. The recorder samples flight state at 10 Hz, including position, velocity, altitude, collision state, and landing status.

Mission outcomes are evaluated against explicit success criteria.

Depending on the scenario, evaluation includes:

- **Mission completion:** Whether the agent achieved the required objective.
- **Flight safety:** Whether the drone avoided collisions.
- **Spatial accuracy:** Whether the drone reached the required position or landing area.
- **Route compliance:** Whether the flight satisfied checkpoints, corridors, and altitude constraints.
- **Decision correctness:** Whether the agent selected the appropriate action for the observed scene.

The visual evaluation framework also reports correct decisions separately from completed missions. This helps distinguish decision-making performance from execution failures.

### Program isolation

In AerialSim-Coding the agent's own box has no simulator. The environment flies practice programs in a separate simulator box and returns what it recorded.

Inside that box the program runs as its own user and reaches the simulator through a proxy that passes only the calls the drone API makes. Calls that would move the drone by hand, reset the world, or change it are refused, and each call is forwarded on its own so a forbidden call cannot ride behind an allowed one. That user is also blocked from the simulator port, and grading refuses to score when the block is not in place; docker practice runs turn that check off with `strict_port` false.

### Reproducibility and validation

Scenarios are generated from predefined seeds, allowing missions to be reproduced across training and evaluation runs.

The environment includes scripted reference pilots and control policies to validate scenario reachability and grading behavior. Held-out evaluation splits are available for testing generalization.

## Drone API

AerialSim-Coding and the original flight environment provide a Python interface for controlling the simulated drone.

```python
from aerialsim_flight.drone import connect

drone = connect()

# Take off to 8 meters
drone.takeoff(height=8)

# Navigate to a position
drone.goto(20, -5, -8, speed=3)

# Observe the environment
frame = drone.camera()

# Inspect the current flight state
state = drone.state()

# Land
drone.land()

# Retrieve the flight result
result = drone.result()
```

The API also supports velocity control, attitude control, angular rates, waypoint navigation, hovering, and camera orientation.

Positions use North-East-Down (NED) coordinates. Flight commands execute synchronously.

## Getting started

Run a basic flight evaluation:

```bash
uv run eval aerialsim-flight -n 3 -r 1 --max-concurrent 1 --no-push
```

The environment starts the simulator, initializes the drone, and executes the selected missions.

AerialSim currently supports one active drone per simulator instance. Use `--max-concurrent 1` when running evaluations.

To select a visual scenario family, use:

```bash
--env.taskset.family obstacle_bypass
```

For held-out evaluation scenarios:

```bash
--env.taskset.split evaluation
```

## Sandbox image

AerialSim uses a container image containing the Cosys-AirSim City Park build, Python dependencies, and drone control interface.

```text
prime/nahidalam/aerialsim-city-park:v3
```

At startup, the container installs the NVIDIA userspace components matching the host driver and launches the simulator in headless mode.

Build scripts are located in `scripts/aerialsim_flight/`.

### Simulator scene

The City Park scene is distributed through the public Hugging Face dataset `nahidalam/aerialsim-scenes`. Building the image does not require a Hugging Face access token.

The build uses a pinned dataset revision and verifies the downloaded archive using SHA-256.

| Field            | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| Dataset revision | `f474f85f3d1b367965ebc77b9b3fa2f88b6f92b0`                         |
| Archive          | `CityParkLite_Linux.tar.gz`                                        |
| Archive size     | 1,372,278,892 bytes                                                |
| SHA-256          | `e84669d338303c180dd3777adb108406248a9f0a774eb102ff3b066a3c1a1772` |

The `build_image.sh` script downloads the specified revision, validates the checksum, and builds the simulator image. A checksum mismatch causes the build to fail.

### Image registry metadata

The following metadata identifies the published image used for the environment.

| Field      | Value                                                                       |
| ---------- | --------------------------------------------------------------------------- |
| Image ID   | `g620vp0zyjx738esnz3gkblz`                                                  |
| Image path | `cmowyasi8001x12c336jqujs6/mnnu0efzhl9gjzm3hihybk8c/linux-amd64/rootfs-cas` |
| Image size | 3,455,645,857 bytes                                                         |
| Published  | 2026-08-30 03:44:55 UTC                                                     |

These values are also recorded in `aerialsim_flight/taskset.py`.

The registry metadata identifies the published image entry. The image tag is mutable and the registry identifiers have not been verified as content hashes. The pinned scene revision and archive checksum provide reproducibility for the simulator scene used during the build.

### Compute requirements

The default task configuration uses:

- 1 NVIDIA RTX PRO 6000 GPU
- 8 CPUs
- 16 GB system memory

The taskset image can be overridden using `--env.taskset.image <ref>`.

## Research applications

AerialSim supports research on:

- Vision-language models for autonomous control
- Reinforcement learning with verifiable mission outcomes
- Visual grounding and spatial reasoning
- Closed-loop planning and decision-making
- Code generation with simulator feedback
- Counterfactual evaluation and failure diagnosis

The environment provides reproducible scenarios, multiple control modes, and trajectory-based verification for evaluating autonomous agents.
