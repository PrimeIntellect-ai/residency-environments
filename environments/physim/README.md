# PhySim

PhySim evaluates an agent's ability to investigate an unfamiliar physical system
and write a predictive model. The agent runs experiments through four laboratory
tools, analyzes the returned observations in a coding harness, and submits
`predict(actions, queries, n_samples=64, seed=0)`. Predictions are graded against
independent retained realizations of the selected world.

## Install and run

From this repository's root, with Python 3.12:

```sh
uv sync --all-extras
uv pip install -e ./environments/physim
uv run --no-sync vf-eval @ configs/physim/eval.toml --model YOUR_MODEL_ID
```

The package pins its Blobkit release, current Verifiers commit, NumPy, and SciPy.
Hugging Face support is included. No local Blobkit checkout or separately selected
extras are needed. The PyPI project named `physim` is unrelated.

The default task is **BF, centered-pulse-v2**, pinned to HF commit
`552229e61813b5684be2051349d778acea054922`. It runs one preparation, never a scan
of the registry. `bf_trail_lab.toml`, `xv_rotor_lab.toml`, and `p4g2_044.toml`
select the three current preparations. A local bundle overrides the BF default:

```sh
uv run --no-sync vf-eval physim --model YOUR_MODEL_ID \
  --env.taskset.task.tools.bundle /absolute/path/to/bundle
```

Alternatively set `env.taskset.task.tools.bundle_source` with `repo`, a full
40-character `revision`, and `path`. Supplying both a local bundle and a remote
source is an error. Every download is hash-verified; `bundle_source.offline=true`
requires a populated cache. See [DATA_SOURCES.md](DATA_SOURCES.md).

## Runtime and grading boundary

The default uses Prime sandboxes and requires Prime credentials and sandbox
credits. Both the agent and predictor use a public, digest-pinned Python image.
Trusted setup installs pinned NumPy, SciPy, scikit-learn, and matplotlib before
network access is restricted. Local image builds are unnecessary.

1. The **trusted evaluator** holds the simulator, exact preparation, grading
   programs, score groups, and truth arrays. A separate trusted MCP process
   exposes only `experiment`, `usage`, `validate`, and `submit`.
2. The **agent sandbox** contains the public interface, scientific Python stack,
   its own workspace, and observations returned by its experiments. It has no
   simulator package, registry, truths, host filesystem mount, or unrestricted
   network access. Model and MCP traffic use Verifiers' permitted framework routes.
3. Submission validates and freezes a bounded regular-file archive. Each grading
   experiment executes that predictor in a **fresh predictor sandbox**, with
   read-only submitted files and observations, a public prediction request, and
   writable temporary storage. That sandbox receives no grading truths, score
   groups, case identifiers, simulator, provider credentials, or network routes.
4. The evaluator collects bounded data-only predictions, destroys the sandbox,
   and compares the frozen arrays with retained truths. Invalid or missing
   predictors earn zero; infrastructure failures remain framework errors.

The task uses Verifiers' public Runtime API for both Docker and Prime. A private
host-only Unix socket connects its MCP process to the task controller for
observation delivery and workspace snapshots. It is never mounted or forwarded
to the agent. Artifact archives are validated without extracting untrusted paths
on the host, preserving Linux case-sensitive filenames on macOS too.

Validation during investigation checks execution, output shapes, finite values,
and seed reproducibility using public interface requests. It provides no physical
accuracy feedback. Successful submission freezes one predictor for the whole
suite; later workspace edits cannot change it.

For local execution, select Docker for both runtimes:

```sh
uv run --no-sync vf-eval @ configs/physim/eval.toml --model YOUR_MODEL_ID \
  --env.agent.runtime.type docker \
  --env.taskset.task.tools.predictor-runtime.type docker
```

The public base image and pinned setup also work locally. Dockerfiles under
`scripts/physim/docker/` are optional preparation aids, not runtime prerequisites.

## Apparatus, score, and limits

Current preparations use `centered-pulse-v2`. Each device's source is centered
on its sensor array. Injection captures the launch position; later adjustment
moves the sensors and the launch point for subsequent injections. Each device
has independent adjustment and injection scheduling. The agent-facing contract
exposes anonymous interfaces and does not disclose these hidden mechanisms.

BF has 4 measurement channels and 15 grading experiments; XV has 6 and 15;
p4g2_044 has 12 and 19. Historical fixed-source bundles retain their original
implementation and truth identities through the frozen compatibility modules.

Let `S` be the mean normalized joint energy across the suite. Reward is
`max(0, min(-log10(S), K)) / K`, with `S=0` mapped to 1. The default precision
is `K=2`; change `env.taskset.task.tools.reward_precision` to set another target.
The requested precision appears in the agent's prompt and grading record.
The scientific energy score is retained separately from reward.

The supplied configs allow 1,000 experiments, 50,000 simulated time units,
128 validations and submissions, 1,024 model turns, and a 24-hour rollout.
These are configurable; tool-budget fields accept `null` in JSON for unlimited
use. Predictor resource limits are separate and disclosed in the prompt.
Costs do not stop a run automatically in the standard `physim` taskset.

## Harnesses and validation

Bash is the default harness. Current Verifiers also supplies `prime-agent`,
Codex, Claude Code, and other harnesses independently of the task.
See [PRIME_AGENT.md](PRIME_AGENT.md) for Prime Agent configuration.

A scripted provider checks the complete experiment, submission, and grading
path without paid inference. It also checks the agent's package, mount,
credential, and network isolation and compares grades with the stored reference:

```sh
uv run --no-sync python scripts/physim/smoke_runtime.py \
  --runtime docker --output outputs/physim-smoke
# Same check on Prime (uses sandbox credits, no paid inference):
uv run --no-sync python scripts/physim/smoke_runtime.py \
  --runtime prime --output outputs/physim-prime-smoke
```

The script runs two independent rollouts by default. Each output directory must
be new. A live model wiring check can use `-n 1 -r 2 --env.agent.max-turns 4`;
that short check is not a capability evaluation.

## Code organization and data contributions

This environment owns the apparatus, experimental interface, taskset, submission
validation, isolation, and grading. Blobkit is the versioned simulation and world
generation library maintained in [swpo/physim](https://github.com/swpo/physim).
The environment does not import the generators directory or the personal
repository. Preparation and validation recipes live in
[generators/physim](../../generators/physim/README.md).

Worlds, preparations, suites, and their provenance live in the
[HF registry](https://huggingface.co/datasets/seanpohorence/physim-worlds).
Contribute through its reviewed pull-request workflow, described in the
[dataset contribution guide](https://huggingface.co/datasets/seanpohorence/physim-worlds/blob/main/CONTRIBUTING.md).
The [project article](https://swpo.github.io/physim/) and
[recorded BF case study](https://github.com/swpo/physim/blob/main/BF_CASE_STUDY.md)
provide scientific context. The recorded study used an earlier harness/runtime;
its reproduction material remains with that study.
