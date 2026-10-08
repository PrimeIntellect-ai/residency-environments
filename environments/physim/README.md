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

The Verifiers pin currently includes the framework fixes proposed in
[Verifiers #2731](https://github.com/PrimeIntellect-ai/verifiers/pull/2731):
output tokens counted once per model call, and native context settings for Pi
and Prime Agent. This is a pinned fork commit pending upstream review.

The default taskset contains **BF, p4g2_044, and XV**, one task per evaluation
preparation in the catalog at pinned HF commit
`bd77a0da2f14eef352bd80c4a38dff426e5c1bed`. Superseded preparations are absent
from this snapshot. Each task is a complete investigation and submission, graded
on that preparation's retained suite. `-n` limits the number of tasks; `-r`
repeats each selected task. For example, `-r 3` runs nine investigations, three
per world. Requesting `-n 50` does not expand this finite three-task set.

The full-eval convention is three tasks with one rollout each. Results retain
`task.data.world_name`, `task.data.bundle_id`, a stable `task.key`, and complete
world/preparation/suite references in `trace.info.r6.references`. Prompts and
apparatus interfaces are rendered separately for each task. These identities
are evaluator metadata, not model-facing descriptions. Equal repetitions give
each world equal weight in the mean reward; report per-world rewards alongside it.

`bf_trail_lab.toml`, `xv_rotor_lab.toml`, and `p4g2_044.toml` select individual
preparations. A local bundle also selects exactly one task:

```sh
uv run --no-sync vf-eval physim --model YOUR_MODEL_ID \
  --env.taskset.task.tools.bundle /absolute/path/to/bundle
```

Alternatively set `env.taskset.task.tools.bundle_source` with `repo`, a full
40-character `revision`, and `path`. Supplying both a local bundle and a remote
source is an error. Every download is hash-verified; `bundle_source.offline=true`
requires a populated cache. See [DATA_SOURCES.md](DATA_SOURCES.md).

For a subset, set `env.taskset.bundles` to a nonempty list of local paths or
remote `{repo, revision, path}` records. Do not combine this list with the
single-bundle options. Duplicate bundle identities are rejected. For example:

```toml
[env.taskset]
id = "physim"
bundles = ["/absolute/path/to/bf-bundle", "/absolute/path/to/xv-bundle"]
```

`env.taskset.catalog` accepts `repo`, `revision`, `cache`, and `offline` to
select another pinned catalog or reuse its verified cache. Publishing new data
does not change an existing run: adopting a new catalog requires changing the
full commit pin. Local paths must exist on each trusted worker; remote sources
are resolved on that worker. Checkpoint recovery requires a single preparation.

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
suite; later workspace edits cannot change it. The harness then finishes its
response naturally. If a run stops before submission, finalization attempts to
collect and validate the last predictor without another model call.

`trace.info.r6.limit_audit` records `submitted_before_finalization` and
`artifact_source` (`agent_submission`, `final_collection`, or null). A collected
predictor is not evidence that the agent submitted it or completed its investigation.
The same record includes the framework stop condition, truncation, provider
length finishes, last-call usage, and any errors observed before grading.

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

Bash is the default harness. Current Verifiers also supplies `prime-agent`, Pi,
Codex, Claude Code, and other harnesses independently of the task. The prompt
names laboratory operations and directs the agent to its harness's advertised
tool names or MCP wrappers.
See [PRIME_AGENT.md](PRIME_AGENT.md) for Prime Agent configuration.

The `qwen-bash.toml`, `qwen-pi.toml`, and `qwen-prime-agent.toml` configs target
the hosted Qwen3.5-9B endpoint's 65,536-token context. They reserve 32,768 tokens
for a response plus 4,096 tokens of tool-result headroom, triggering native
compaction around 28,672 tokens. Pi and Prime Agent retain 12,000 recent tokens
when compacting. Context limits belong to the model/provider pairing; verify
them before adapting these profiles to another endpoint. Total rollout token
budgets are separate from the per-request context window.

A scripted provider checks the complete experiment, submission, and grading
path without paid inference. It also checks the agent's package, mount,
credential, and network isolation and compares grades with the stored reference:

```sh
uv run --no-sync python scripts/physim/smoke_runtime.py \
  --runtime docker --output outputs/physim-smoke
# Same check on Prime (uses sandbox credits, no paid inference):
uv run --no-sync python scripts/physim/smoke_runtime.py \
  --runtime prime --output outputs/physim-prime-smoke
# Exercise finalization's automatic collection instead of an explicit submission:
uv run --no-sync python scripts/physim/smoke_runtime.py \
  --runtime docker --completion collect --output outputs/physim-collection-smoke
```

The script runs two independent rollouts per world by default (six total).
Pass `--bundle /absolute/path/to/bundle` for a single-world check. Each output
directory must be new. A live model wiring check can use `-n 1 -r 2 --env.agent.max-turns 4`;
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
