# Live integration smoke test

Validated on 2026-09-03 with Verifiers 0.3.1 and Docker 28.3.3. This is a
functionality check on one prompt with two responses, not a performance
estimate or judge-quality evaluation. A subsequent review change removed
unused biological gold columns from the compound table without changing any
tool-consumed field or output; regeneration, tool-parity, and package checks
were rerun after that change.

## Configuration

```bash
uv run eval @ configs/drug-perturbation/eval.toml \
  @ configs/drug-perturbation/process-reward.toml \
  -n 1 -r 2 -c 2 -m qwen/qwen3.5-35b-a3b --no-push
```

The policy was the available base-model ID, not a trained adapter. The judge
was `google/gemini-3-flash-preview`, with the unchanged
`reasoning-process-v0.3` rubric, temperature 0, and 900 output tokens.
Policy sampling temperature was not overridden; the output limit was 16,384
tokens with at most five turns. The default test task selection and seed 42
used the pinned Hugging Face dataset revision
`49d20f9f293fea3f4c93ab57b478aaa3f1651d84`.

## Observed results

| Response | Deterministic D | Process J | D × J |
| --- | ---: | ---: | ---: |
| 1 | 0.675000 | 0.908710 | 0.613379 |
| 2 | 0.600000 | 1.000000 | 0.600000 |

Both responses completed normally in two model turns, each with one
`identify_compound` call. Both had complete answer tags and no rollout errors
or token-limit termination. Both judge calls produced parseable verdicts with
valid evidence locations on the first attempt; neither failed open. Judge
latencies were 5.47 and 4.47 seconds. Reported usage cost was $0.0037 for the
policy and $0.0087 for the judge ($0.0124 combined; provider-reported values).

Local rescoring of these same traces under the default D-only configuration
returned the original D values, made zero judge calls, and left J unavailable.
The temporary evaluation workers and containers were cleaned up.

The observed attenuation verifies reward wiring, not whether the judge's
biological assessment is correct. This does not establish Hosted Lab training
compatibility, alternative-harness isolation, or stochastic trajectory parity
with earlier environment versions. Raw traces are not bundled with this
environment contribution.

## Five-pathway limit

The pathway reward now becomes zero when the answer lists more than five
entries, matching the prompt's requested count. The historical signed F1
remains a raw metric, alongside the applied pathway score, entry count, and
overflow flag. This is an explicit scoring change from the original port.

Local validation covered 11 cases: empty, four, five, and six entries;
duplicate, malformed, invalid-direction, and opposite-direction sixth
entries; semicolon and newline separators; and blank separators. Native
task scoring confirmed that other components retain credit and that the
new D feeds both the default reward and optional D×J mixture. The 21 original
scoring fixtures, 32 archived replays, and two recorded live responses retained
their earlier scores. Lint, formatting, and both repository package checks
also passed. No new model generations were needed for these scoring checks.

## Per-example target limit

The target reward now becomes zero above twice the number of unique,
case-normalized gold targets for that example. Supplied entries are counted
before deduplication; no list is truncated. Raw target F1 remains available
alongside the applied target score, entry count, per-example limit, and
overflow flag. This is the second explicit scoring change from the original
port. Both caps apply within D, which also feeds D×J.

Local validation covered 15 target cases: missing/empty answers, exact
answers, the limit and first overflowing entry, duplicate/malformed extras,
comma/pipe and blank separators, one-target references, duplicate/case-varied
gold entries, partial matches below the limit, absent gold, and a synthetic
201-entry answer with two of three gold targets. The last case retains a
positive raw F1 but receives zero target reward.

Six native reward-mixture checks covered target overflow alone and combined
target/pathway overflow; other biological components retained their credit.
An unrequested-target case confirmed null target metrics and unchanged
reward. Zero-weight judge skipping still worked. All 11 pathway boundary
cases, 21 original scoring fixtures, 32 archived replays, two tool fixtures,
and both recorded live responses passed rescoring without changing their
earlier scores. Lint, formatting, and both repository package checks passed.
No new model generations were made. These are scoring/integration checks,
not evidence that the caps eliminate all overprediction or improve learning.

## Global answer-format gate and numeric parsing

The whole deterministic reward is now zero unless every requested answer tag
has exactly one nonempty, well-formed block in the final reply. Required
viability values must be a single finite number, not a number extracted from
prose or a placeholder. D×J inherits the same gate. The loader appends the
explicit format contract to each prompt and hashes the actual augmented
prompt; source data and source keys are unchanged.

Local checks covered 61 structural rejection cases, 11 accepted numeric
forms, 20 rejected numeric forms, and 24 native reward-mixture cases. These
include case-varied duplicates, unmatched/partial delimiters, nested blocks,
empty/missing fields, `-0.X`, multiple numbers, NaN/infinity, numeric overflow,
leading-decimal and scientific notation, task-specific required fields, and
zero-weight judge skipping. Earlier assistant turns and separate reasoning
content do not contaminate final-answer validation. Simplified reproductions
of both reported parser exploits now receive zero reward.

All 186,854 train and 20,257 test rows loaded with unique source keys and
correct augmented prompt hashes. Gold-formatted examples for all 33 selected
task-view/phenotype combinations earned full credit; removing any required
block gave zero. The target/pathway cap checks and both tool fixtures passed.
Twenty of the 21 historical scoring fixtures remain unchanged. The fixture
that intentionally accepted a number embedded in prose now receives zero,
as required by the new numeric contract. All 32 archived response scores and
both saved live response scores remain unchanged. Lint, formatting, and both
repository package checks passed.

No new model generations were made. The saved live traces used the earlier
prompt and have only been rescored; the new prompt wording has not received
a fresh live model evaluation. These checks establish local parser/reward
behavior, not training improvements or format-compliance rates.
