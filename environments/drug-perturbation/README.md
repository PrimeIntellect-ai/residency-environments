# Drug perturbation

Verifiers v1 tasks for predicting a small molecule's targets, mechanism of
action, signed Hallmark pathways, and phenotype in a cancer cell line.

Full-chain prediction from SMILES is the default. Other task views and a
continuous process judge are opt-in. The package includes no training runs,
research reports, or privileged-evidence agentic judge.

## Installation and evaluation

From the repository root:

```bash
uv sync
uv pip install -e ./environments/drug-perturbation
uv run eval @ configs/drug-perturbation/eval.toml -m <model-id> -n 1 -r 2
```

The example uses the built-in `null` harness (native tool calling) in Docker.
Docker must be running. Model requests use Verifiers' configured inference
endpoint; this does not run model weights on the local machine. Remove `-n 1
-r 2` for the standard 100-prompt, eight-rollout evaluation. The config does
not override policy sampling temperature and disables automatic result upload.

No custom harness is required. A containerized harness with supported network
controls is required: the task requests framework-only network access. Do not
mount evaluator directories, Hugging Face caches, reference answers, or
research artifacts into the solver's container. The provided configuration
does not mount those files. Alternative harnesses are different benchmark
conditions and should be validated separately. A Prime runtime needs its
network-policy-capable VM configuration; local Docker validation does not
establish Hosted Lab compatibility.

## Task views

All views share one taskset and the same compound-disjoint split. Select them
with `env.taskset.entry_points`; the default is `["smiles_only"]`.

| Entry point | Supplied information beyond SMILES/context | Requested outputs |
| --- | --- | --- |
| `smiles_only` | None | Target, mechanism, pathways, phenotype |
| `from_target` | Target | Mechanism, pathways, phenotype |
| `from_moa` | Target and mechanism | Pathways, phenotype |
| `from_pathways` | Target, mechanism, pathways | Phenotype |
| `phenotype_direct` | None | Phenotype |
| `target_from_smiles` | None | Target |
| `moa_from_smiles` | None | Mechanism |
| `moa_from_target` | Target | Mechanism |
| `pathways_from_smiles` | None | Pathways |
| `pathways_from_moa` | Target and mechanism | Pathways |
| `phenotype_from_moa` | Target and mechanism | Phenotype |
| `phenotype_from_pathways` | Pathways | Phenotype |

`phenotypes` defaults to `viability`, `cell_cycle`, and `stress`; `magnitude`
is optional. This filter only restricts phenotype questions, not upstream
questions. Optional `cell_lines` restricts the cell contexts. `split` defaults
to `test`; training should explicitly select `train`. `seed=42` determines
the order before the evaluation CLI takes its requested number of tasks.

Selecting several views concatenates their eligible rows and shuffles them;
it does not balance their frequencies. To control task proportions, configure
separate named training sources using the same taskset with different
`entry_points` and source `ratio` values. A four-source example is in
`configs/drug-perturbation/task-mixture.toml`; it is an overlay, not a complete
training recipe. Sampling weights do not guarantee the same proportions after
zero-advantage filtering. A curriculum is a sequence of runs with different
task selections, not a hidden schedule in the environment.

## Deterministic reward

Reference answers contain only the components requested by each prompt.

| Component | Metric | Default weight |
| --- | --- | ---: |
| Target | Gene-symbol set F1 | 0.15 |
| Mechanism | Case/whitespace-normalized exact match | 0.15 |
| Pathways | Set F1 of signed Hallmark-name/direction pairs | 0.25 |
| Phenotype | Viability tolerance score or categorical accuracy | 0.45 |

The deterministic reward `D` is the weighted mean over applicable components,
renormalized by their weights. An isolated pathway task therefore receives its
pathway F1, not 0.25 times that value. Component weights can be set through
`env.taskset.task.component_weights`; selecting a task whose applicable
components all have zero weight is an error.

For viability, absolute log2-fold-change error at most 0.25 earns 1; error at
least 2 earns 0; intermediate errors are linearly interpolated. The categorical
phenotypes use exact class accuracy. Signed pathway F1 ignores the
`HALLMARK_` prefix and case; the separate pathway-name metrics use the known
Hallmark vocabulary. Missing or malformed requested answers receive zero for
that component. Format compliance is diagnostic, not an additional reward.

Raw metrics include `deterministic_reward`, `target_f1`, `moa_accuracy`,
`pathway_signed_f1`, `pathway_name_validity`, `pathway_name_f1`,
`pathway_direction_accuracy`, `phenotype_score`, and `format_compliance`.
Unrequested components are recorded as null, with explicit `*_applicable`
indicators. Do not interpret null as an incorrect answer. Task-balanced
reporting is an analysis aggregation, separate from these per-response scores.

## Optional process reward

The built-in reward components are `deterministic_reward = D` and
`process_adjusted_reward = D * J`. For nonnegative component weights `[a,b]`
with `a+b=1`, the training reward is `a*D + b*D*J`, always between zero and D.
These weights are different from the biological component weights above.

```toml
[env.taskset.task.rewards]
deterministic_reward = { weight = 1.0 }
process_adjusted_reward = { weight = 0.0 }
```

This is the default: **J is not computed when its coefficient is zero**. No
judge credentials or model requests are needed. `process_judge_computed=0`
and null judge/adjusted-score metrics distinguish skipped computation from
a real score of zero or one. Set `[0,1]` for full dampening or `[0.5,0.5]`
for half-strength dampening. For example:

```bash
uv run eval @ configs/drug-perturbation/eval.toml \
  @ configs/drug-perturbation/process-reward.toml -m <policy-model-id> -n 1 -r 2
```

The process judge receives the visible prompt and full linear solver trace,
including exposed reasoning and tool results, **not the reference answers**.
The versioned `reasoning-process-v0.3` rubric labels reasoning nodes as
supported/weak/unsupported/contradicted and transitions as
coherent/weak/disconnected/contradicted. Fixed label values 1/0.65/0.25/0
are combined by geometric mean. Cited message locations must exist and contain
text; a generated node or transition cannot earn high support from bare answer
tags alone. These checks validate references, not biological truth or semantic
entailment. This judge is a process heuristic, not a factual oracle.

The tested rubric is available for `smiles_only` and the four direct links
`target_from_smiles`, `moa_from_target`, `pathways_from_moa`, and
`phenotype_from_pathways`, with the three standard phenotypes. Other views
and magnitude remain available with deterministic-only reward; unsupported
judge/task combinations fail validation instead of silently bypassing judging.
Branching/compacted solver traces are not supported by this linear-trace rubric.

The historical judge model is `google/gemini-3-flash-preview`; availability
depends on the configured provider. Override
`env.taskset.task.process_judge.model` for other models, recording the choice
as a changed experimental condition. The endpoint/key-variable fields follow
Verifiers `JudgeConfig`. Defaults are temperature 0, 900 output tokens, two
retries for transient transport errors, and 180 seconds per attempt. No retry
is made to improve a semantically undesirable verdict or malformed verdict.

Transport, parse, or reference-validation failure intentionally fails open
with effective J=1, preserving deterministic reward. Separate failure flags,
raw judge output, label fractions, latency, retries, and available usage/cost
are recorded. Exclude fail-open verdicts when analyzing judge quality. A
failure-dominated run is not evidence that the process judge has no effect.
The judge runs in the evaluator, outside the solver's network restrictions.

## Compound lookup

The optional `identify_compound(smiles, top_k=5)` tool returns a name match,
molecular descriptors, scaffold, and chemically similar compound names.
It returns no targets, mechanisms, pathways, expression values, or phenotype
labels. Its packaged lookup table physically contains only SMILES, InChIKey,
and compound name, and the loader rejects any broader schema. It is enabled by
default to match the tool-assisted task; set
`env.taskset.task.tools_enabled=false` for the no-tool condition. `top_k` is
bounded to 1–20. The narrowly scoped tool provides identity/chemistry without
opening access to biological answer databases. No Hallmark or gene-expression
lookup tool is exposed.

## Dataset and provenance

The examples are hosted at
[seanpohorence/drug-perturbation](https://huggingface.co/datasets/seanpohorence/drug-perturbation).
The loader pins revision `49d20f9f293fea3f4c93ab57b478aaa3f1651d84` and verifies
the selected file's SHA-256. There are 186,854 train and 20,257 test rows across
all views/phenotypes. The default test selection has 1,834 eligible rows, of
which the standard evaluation takes 100. Reference tables are not shipped in
the environment wheel or copied into the solver runtime.

`source_row` is the row position within the frozen split, before filtering or
shuffling. Stable task keys include it to preserve duplicate-row multiplicity;
`prompt_key` groups identical visible prompts. Do not treat multiple views or
duplicate rows as independent biological observations. A checksummed local
copy of the original public table can be supplied via `dataset_path` for
offline validation. The export helper is under `scripts/drug-perturbation/`.

This port preserves deterministic scoring from the public
[drug-perturbation-rl](https://github.com/swpo/drug-perturbation-rl) environment
and the versioned continuous process rubric. It removes the old role prompt,
uses current Verifiers tasksets/toolsets, adds container/network restrictions,
and bounds the optional neighbor count. Consequently, scoring parity does not
claim byte-identical complete prompts, identical tool schemas, Hosted Lab
compatibility, or equivalent stochastic RL trajectories.

A one-prompt/two-response live Docker smoke test with the Qwen3.5-35B base
model and Gemini process judge completed successfully; see the
[validation note](../../scripts/drug-perturbation/VALIDATION.md) for the exact
configuration, reward checks, and limitations. This is an integration check,
not an estimate of model or judge performance.

The code is Apache-2.0. Source data retain their own terms and attribution:
see [DATA_SOURCES.md](DATA_SOURCES.md). This is a research benchmark, not
clinical treatment guidance.
