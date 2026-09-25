# Preparing a world for science and evaluation

A preserved genome becomes eval-ready only for an exact preparation, apparatus,
suite and independently simulated truth. The BF and XV recipes establish this
workflow; they are concrete scientific recipes, not automatic certification of
arbitrary genomes.

1. **Simulate afresh and preserve the origin.** Inspect full activator and channel
   fields with the current phenomenology. Keep initialization, seed, numerical
   settings, source hashes and the resulting fields. BF starts from its first
   2,500-tu soup continuation; XV starts from a prepared two-organism pair evolved
   for 2,500 tu without a kick. A fixed horizon does not establish convergence.
2. **Prepare a measurable laboratory.** Preserve exact fields and position probes
   by a declared rule. Here, the activator-0 maximum sets both probe centers and
   the source sits six units away. Expose only anonymous ports and observations.
3. **Demonstrate ordinary interventions.** Compare no-source observations, weak
   and strong pulses, different channels, delayed/composed pulses, translation
   and dilation. Use three independent future-noise realizations per program.
   Inspect spatial fields as well as sensor arrays. Bands show observed ranges,
   not confidence intervals.
4. **Check the physical interpretation.** Use explicitly privileged mechanism
   controls separately from agent-accessible actions. BF removes its bilinear
   trail feedback; XV removes cross-drive source weights. Do not claim that a
   model knows these mechanisms merely because its predictor scores well.
5. **Freeze the suite before grading truth.** These suites have 11 programs,
   four physically scaled score groups, a 50-tu horizon and 64 forecast members.
   Generate two new truths per case with seeds separate from development and
   diagnostic forecasts. Changing the suite or preparation creates a new bundle.
6. **Check score meaning and the actual environment.** Compare independent native
   forecasts, ignored actions, removed feedback, wrong probe geometry and initial
   persistence. Check the native scheduler, submission interface, reference score,
   bundle identities and one bounded stock Verifiers rollout. These native
   controls are privileged diagnostics; they are not learned-agent performance.
7. **Preserve and expose the completed result.** Archive the full runnable bundle,
   source fields, recipes, run metadata and evidence in the registry. Reconstruct
   and validate the bundle from registry bytes, then mark the new preparation
   eval-ready. Keep the original preserved genome/world records immutable.

## Commands

Use Python 3.12. From the repository root, install the environment with
`uv pip install -e './environments/physim[reference,hub]'`. This installs the pinned
public Blobkit wheel and reference numerical dependencies. Each output directory
must be new. No plotting package or personal research checkout is required.

Fetch the preserved simulation endpoint and provenance from the public registry:

```sh
uv run python generators/physim/fetch_inputs.py \
  --repo seanpohorence/physim-worlds \
  --revision dcd6abd5eae76a47f326c70518315d2d1e101d86 \
  --world bf_trail_lab --output outputs/bf-inputs
```

Use `xv_rotor_lab` for XV. Inputs contain `genome.json`, `final_state.npz`, the
archived upstream protocol, and a download receipt with the dataset commit and
registry references. Downloads and verification do not execute archived source.
Add `--offline` to reuse the HF cache. These recipes start from the preserved
2,500-tu endpoint; they do not rerun the historical search or initial simulation.
They generate fresh experimental continuations and grading truths from that state.

```sh
uv run python generators/physim/eval_preparation.py --world bf \
  --source outputs/bf-inputs --output outputs/bf-new
uv run python generators/physim/bf_feedback_check.py \
  --preparation outputs/bf-new/preparation --output outputs/bf-new/causal
uv run python generators/physim/build_evaluation_bundle.py \
  --source outputs/bf-new --output outputs/bf-new/bundle
uv run python generators/physim/validate_evaluation.py --source outputs/bf-new
```

For XV use `--world xv` and `xv_feedback_check.py`; its optional `--horizon 250`
extends the mechanism control beyond the 50-tu agent contract. The builder
uses three CPU workers by default. Full science and truth generation takes
minutes; no GPU is required. Inspect observations and controls before treating
a replay as scientific validation of a new preparation.

After building the agent and predictor images described in the environment README,
run a bounded model smoke against the new local bundle:

```sh
uv run eval @ configs/physim/eval.toml --model YOUR_MODEL_ID \
  --env.taskset.task.tools.bundle "$PWD/outputs/bf-new/bundle" \
  -n 1 -r 2 --env.agent.max-turns 4
```

This makes paid model calls and tests wiring; four turns do not establish learned
physics. Use the generous full config for a scientific rollout. The preparation,
genome and fields stay on the trusted host; the agent and predictor receive only
their contract, observations, and own files.

After native validation passes and the scientific evidence has been reviewed:

```sh
uv run python generators/physim/register_evaluation.py add --source outputs/bf-new \
  --name bf_replay --registry outputs/replay-registry
uv run python generators/physim/export_registry_catalog.py --registry outputs/replay-registry
uv run python generators/physim/register_evaluation.py export WORLD_RECORD_ID \
  --registry outputs/replay-registry --output outputs/restored-bundle
uv run blobkit registry export-recipe outputs/replay-registry RECIPE_ID outputs/restored-recipe
```

Use the world and recipe IDs printed by registration. If including a
`pilot_summary.json`, pass `--pilot-config` with its exact run config; additional
executed source archives can be supplied with repeated `--source-archive` paths.
Registration writes a local registry. Publishing it is a separate reviewed
[HF contribution](https://huggingface.co/datasets/seanpohorence/physim-worlds/blob/main/CONTRIBUTING.md);
neither registration nor publication automatically selects it for evaluation.

Export never executes archived Python. The exported recipe includes its precise
input fields and source versions; replay creates new data files. The exported
bundle preserves the original bytes and identity. The first `p4g2_044` bundle is
an export of historical prepared fields and retained truths: use its published
bundle for exact reproduction. BF/XV are the portable fresh-truth recipes here.

`build_evaluation_bundle.py --resume` only resumes an incomplete build whose
preparation and frozen suite are unchanged. It reuses already written truth and
forecast files; completed bundles cannot be overwritten. Preserve the original
and repaired executed-source versions and record any recovery in the report.

## Scope of the first two preparations

BF has four ports and a stationary trail channel (diffusion zero); XV has six
ports. The environment derives port count from validated genome dimensions and
keeps the same scheduler, source/probe semantics and scoring kernels as the
12-port reference. Supported genomes remain bounded to the documented simulator
schema. A single preparation, two grading truths and one model pilot are useful
development evidence, not a generalization or model-comparison benchmark.
