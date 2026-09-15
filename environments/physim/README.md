# Physim

A Verifiers v1 taskset for learning unfamiliar physical systems through experiments.
An agent uses a coding harness to collect observations, then submits
`predict(actions, queries, n_samples=64, seed=0)`. The host grades coherent sample
trajectories against independent retained physical realizations.

## Installation

Use Python 3.12 and install the environment from the repository root:

```sh
uv pip install -e './environments/physim[reference,hub]'
```

Physim 0.12.2 depends on blobkit 0.3.5 and Verifiers 0.3.1–0.3.x. The `reference`
extra pins NumPy 2.5.2 and SciPy 1.18.0; `hub` adds immutable Hugging Face downloads.
The empty `agent` extra is retained for old installation commands.

The standalone distributions are published as GitHub release assets:
[Blobkit 0.3.5](https://github.com/swpo/physim/releases/tag/blobkit-v0.3.5) and
[Physim 0.12.2](https://github.com/swpo/physim/releases/tag/physim-v0.12.2).
Physim declares the Blobkit wheel URL with its SHA-256, so installation does not
need `packages/blobkit/` or the uv workspace. The PyPI project named `physim` is
unrelated; use this repository's distribution explicitly.

For a fresh installation, download `requirements.txt` and `physim-setup.tar.gz`
from the Physim release. The requirements file pins the Physim wheel by SHA-256;
the setup archive contains the world configs, Dockerfiles, examples, and release
verification helper. Its checksum is in `SHA256SUMS` on the same release.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

The prepared data are published and have passed anonymous download, offline-cache,
and native checks. The Dockerfiles can be built by a reviewer; publishing prebuilt
images is optional. The agent image is `physim-agent:0.12.2`; it caches the
stock Verifiers bash-harness dependencies during the build and starts offline.
The predictor image remains `physim-predictor:0.12.0` with unchanged numerical
dependencies. Rebuild the agent image when upgrading the host Verifiers version.

## Data and setup

Select a prepared world using a published-world config below, or supply a verified
local bundle with `env.taskset.task.tools.bundle`. **There is no default world**, and the environment does not automatically select eval-ready
registry entries. Omitting the bundle fails before model execution with an error
explaining how to select one. Supplying both a local bundle and `bundle_source`
is an error. The first bundle contains one prepared
`p4g2_044` world and 15 grading programs with two retained truths each. The package
contains no private physics arrays or grading truths. See [DATA_SOURCES.md](DATA_SOURCES.md).

The bundle determines the public port count: BF uses four, XV six, and the
original reference twelve. Each uses the same 13-node and 19-node probes, global
mean/variance readings, 50-tu contract and native scheduler. Stationary channels
may have zero diffusion. See the [archived preparation recipes](https://huggingface.co/datasets/seanpohorence/physim-worlds/tree/dcd6abd5eae76a47f326c70518315d2d1e101d86/registry)
for the scientific checks, independent truth generation and registry export.

From the repository root, build the isolated predictor and agent images:

```sh
docker build -f scripts/physim/docker/predictor.Dockerfile \
  -t physim-predictor:0.12.0 scripts/physim/docker
docker build -f scripts/physim/docker/agent.Dockerfile \
  -t physim-agent:0.12.2 scripts/physim/docker
```

The standard taskset ID is `physim`. It exports one native taskset class,
`PhysimTaskset`. The old `physim_r6` ID remains an alias; `physim_r6_scaling` retains
an optional dollar-budget variant.

## Published world selection

Each config selects exactly one preparation at HF commit
`dcd6abd5eae76a47f326c70518315d2d1e101d86`:

| Config under `configs/physim/` | Preparation | Public ports |
| --- | --- | ---: |
| `p4g2_044.toml` | Original reference | 12 |
| `bf_trail_lab.toml` | BF trail lab | 4 |
| `xv_rotor_lab.toml` | XV rotor lab | 6 |

The `env.taskset.task.tools.bundle_source` block names the repository, full commit,
and bundle path. The trusted host downloads and verifies the selected data before
model execution. It never mounts the bundle into the agent container. `cache`
optionally selects a cache directory; `offline = true` requires an existing
verified cache. Only the selected preparation runs; to evaluate the fixed set,
run the three configs separately and report each result.

```sh
uv run eval @ configs/physim/bf_trail_lab.toml -m YOUR_MODEL_ID --dry-run
# Initial wiring smoke; four turns are not a capability evaluation.
uv run eval @ configs/physim/bf_trail_lab.toml -m YOUR_MODEL_ID \
  -n 1 -r 2 --env.agent.max-turns 4
# Full investigation uses the generous limits in the selected config.
uv run eval @ configs/physim/bf_trail_lab.toml -m YOUR_MODEL_ID
```

Use the corresponding XV or original-reference config to select another world.
For a local bundle use `eval.toml` and the bundle argument shown below.

## Evaluation

An otherwise completed rollout with no predictor or an invalid predictor earns
zero reward and records `no_predictor` or `invalid_predictor`. Infrastructure
failures remain native Verifiers errors. A four-turn smoke may therefore finish
with zero reward while still confirming that the environment is wired correctly.

Resolve configuration without calling a model (`--dry-run` does not load or
verify the bundle):

```sh
uv run eval @ configs/physim/eval.toml \
  --env.taskset.task.tools.bundle /path/to/reference-bundle --dry-run
```

With provider credentials configured, start with Prime's one-example, two-rollout
smoke convention. This command incurs model API costs:

```sh
uv run eval @ configs/physim/eval.toml -m YOUR_MODEL_ID \
  --env.taskset.task.tools.bundle /path/to/reference-bundle \
  -n 1 -r 2 --env.agent.max-turns 4
```

For a full investigation, omit those smoke overrides. The full-eval convention is
one preparation × one rollout; each accepted predictor is evaluated across all
programs in its bundle (15 in the original reference, 11 in BF and XV). The
four-turn smoke checks integration, not forecast quality.

The selected bundle yields one task. Increasing `num_rollouts` repeats independent
investigations of that same preparation; increasing `num_tasks` does not select
more worlds. To evaluate another prepared world, run again with its bundle path.
The registry's eval-ready status records readiness, not inclusion in an eval run.

### Exploration budgets

The default eval config aims to give models room to investigate, with no efficiency
penalty in the reward:

| Limit per rollout | Default |
| --- | ---: |
| Model turns | 1,024 |
| Total generated output tokens | 1,048,576 |
| Agent elapsed time | 24 hours |
| Laboratory experiments | 1,000 |
| Integrated simulation time | 50,000 tu |
| Public validations / submission attempts | 128 each |
| Agent container CPU / memory | 4 cores / 8 GB |

These are ceilings, not required usage. A successful submission ends the
investigation early. The turn, token, timeout, and runtime settings use Verifiers'
native config; laboratory limits are enforced by Physim's experiment service.
The task's 24-hour fallback is overridden by `env.agent.timeout.rollout`.

For a longer investigation, increase the relevant limits together, for example:

```sh
uv run eval @ configs/physim/eval.toml -m YOUR_MODEL_ID \
  --env.taskset.task.tools.bundle /path/to/bundle \
  --env.agent.max-turns 4096 --env.agent.max-output-tokens 4194304 \
  --env.agent.timeout.rollout 172800 \
  --env.taskset.task.tools.max-experiments 5000 \
  --env.taskset.task.tools.max-total-tu 250000 \
  --env.taskset.task.tools.max-validation-attempts 512 \
  --env.taskset.task.tools.max-submission-attempts 512
```

Omitting a turn or token limit from a copied config leaves that native Verifiers
limit unset. Model context windows and provider limits still apply. The experiment
service supports up to 1,000,000 total tu. Each independent experiment retains its
50-tu physical horizon.

These settings govern exploration. The submitted predictor currently runs under
separate grading limits: one CPU, 1 GB, and 30 seconds per prediction. Increasing
the exploration budget does not change that execution contract.

### Completion and errors

Use Verifiers' native `trace.stop_condition`, `trace.is_truncated`, `trace.ok`, and
`trace.errors` for execution status. Verifiers classifies provider, harness,
sandbox, toolset, and task errors and provides configurable agent/episode retries
and eval resume. A turn/token stop is recorded differently from a wall-clock
timeout; the latter is a harness error in the installed Verifiers 0.3.1 runtime.
The shipped config disables whole-run retries; SDK-level transient retries are
separate. Physim adds only task-specific submission and grading information.

Physim uses Verifiers' stock bash harness and Docker runtime. Its narrow laboratory
tools run on the trusted host because experiment budgets, hidden physical state,
and immutable submissions require a boundary outside the agent's filesystem.
There is no custom model loop. The agent has no network access by default.

## Reward and reference checks

Lower joint energy is better. The task maps it to Verifiers reward as
`1 / (1 + primary_joint_energy)`. An otherwise completed rollout with a missing
or invalid predictor receives zero reward. Infrastructure failures remain native
Verifiers errors.
The disclosed reference world does not measure unfamiliar-world generalization.

```sh
uv run physim demo --bundle /path/to/reference-bundle --output outputs/demo
```

This model-free persistence control reproduces `0.8271209896216252` with four
forecast members. Model evaluation uses 64. The original scheduler, experiment
service, scoring code, and CPU kernels retain their scientific identities.

## Licensing

Physim and blobkit code are Apache-2.0; the reference data are CC-BY-4.0. The chosen
dataset is [seanpohorence/physim-worlds](https://huggingface.co/datasets/seanpohorence/physim-worlds/tree/dcd6abd5eae76a47f326c70518315d2d1e101d86), published and verified at
`dcd6abd5eae76a47f326c70518315d2d1e101d86`. See [DATA_SOURCES.md](DATA_SOURCES.md)
for scope and the personal project's [release instructions](https://github.com/swpo/physim/blob/main/RELEASING.md).

## Prepare and contribute worlds

The [portable preparation workflows](../../generators/physim/README.md) use the
published HF registry and installed Blobkit to reproduce BF/XV preparations, run
fresh simulations, freeze suites, generate truth, and validate runnable bundles.
Research and general Blobkit development live in the
[personal project](https://github.com/swpo/physim); published world data and
provenance live on [HF](https://huggingface.co/datasets/seanpohorence/physim-worlds).
