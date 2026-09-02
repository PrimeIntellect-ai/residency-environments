# Eleusis

Eleusis is a long-horizon inductive-reasoning environment. A hidden rule
determines whether each card is accepted or rejected. The model experiments by
playing cards and submits its current rule hypothesis on every turn.

Rules are loaded from an external rule bank: a Hugging Face dataset or a local
`Dataset`/`DatasetDict`. The default bank is
[`nph4rd/eleusis-calibrated`](https://huggingface.co/datasets/nph4rd/eleusis-calibrated):
2,048 rules in `train` and 512 in `test`, balanced across eight rule families,
one deal per rule by default, and up to 100 valid plays per episode. The
environment pins dataset revision
`f4d1aeef1617df8ac30454dd2884a67d3e7d0d93`; setting a different dataset or
explicit revision remains supported.

The default dataset is protocol v0.7.2. Every test rule passes deterministic
reachable-trajectory and information-gain gates. The test split contains 512
distinct measured behaviors and has no measured behavioral overlap with train.
Rules are interleaved by family when loaded, so evaluation and training prefixes
remain representative instead of traversing one family at a time.

## How it works

Each turn the model calls `play(rule, card)`:

- `card` must be a card in its current hand.
- `rule` is a Python boolean expression over `card` and `mainline`.
- The card is accepted or rejected by the hidden rule.
- The hypothesis is checked by deterministic behavioral equivalence.

The episode ends when the hypothesis is correct or after 100 valid plays.
Solved episodes receive earlier-is-better reward:

```text
reward = (101 - turns_used) / 100
```

Unsolved episodes receive zero.

## Run

```bash
uv run eval eleusis -m <model>
```

Common options (all optional; env knobs go under `--env.taskset.*`):

- `--env.taskset.dataset <owner/rules>` — rule bank: Hub repo or local path.
- `--env.taskset.split <name>` — split to evaluate (default `test`).
- `--env.taskset.max-turns <n>` — valid plays per episode (default 100).
- `--env.taskset.hand-size <n>` — initial hand size (default 12).
- `--env.taskset.rounds-per-rule <n>` — seeded deals per rule (default 1).
- `--env.taskset.seed <n>` — deal seed (default 20260812).
- `--env.taskset.revision <commit>` — override the pinned Hub revision.
- `--sampling.max-tokens <n>` — completion cap per model call.
- `--sampling.temperature <x>` — temperature (left at the model default unless
  set).

## Design notes

- **Custom tool:** the `play(rule, card)` tool is the environment. The game
  state machine must intercept every action, evaluate the hidden rule, update
  the board, and check the hypothesis by behavioral equivalence — this cannot
  be reduced to a plain harness task, which is why Eleusis ships a toolset
  rather than a bare prompt taskset.
- **Isolation:** rollout network access is disabled because the public rule bank
  contains the answer code. Model-supplied hypotheses run in a reusable child
  process with CPU, memory, and wall-time limits; an over-limit hypothesis is
  incorrect and the game continues. A hypothesis must agree with prior verdicts
  from the rollout and a deterministic probe battery spanning the full game
  horizon before it is accepted.
- **Prompt:** the system prompt documents the game rules, the expression
  language, and the scoring contract, and is kept stable because benchmark
  numbers are measured under it; it intentionally does not teach play strategy.

## Rule dataset format

A rule source must contain unique, non-empty `rule_id` and `code` columns.
`code` is a Python boolean expression (or function body) over `card` and
`mainline` that returns whether a candidate card is accepted, for example:

```python
return card.color == "red"
```

Optional source control: `--env.taskset.dataset-config <name>` selects a
dataset configuration within the rule bank.
