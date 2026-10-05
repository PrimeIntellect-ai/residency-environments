# Eleusis

Eleusis is a long-horizon inductive-reasoning environment. A hidden rule determines whether each card is accepted or rejected. The model experiments by playing cards and submits its current rule hypothesis on every turn.

Rules are loaded from an external rule bank: a Hugging Face dataset or a local `Dataset`/`DatasetDict`. The default bank is [`nph4rd/eleusis-calibrated`](https://huggingface.co/datasets/nph4rd/eleusis-calibrated): 2,048 rules in `train` and 512 in `test`, balanced across eight rule families, one deal per rule by default, and up to 100 valid plays per episode. The environment pins dataset revision `f4d1aeef1617df8ac30454dd2884a67d3e7d0d93`; setting a different dataset or explicit revision remains supported.

The default dataset is protocol v0.7.2. Every test rule passes deterministic reachable-trajectory and information-gain gates. The test split contains 512 distinct measured behaviors and has no measured behavioral overlap with train. Rules are interleaved by family when loaded, so evaluation and training prefixes remain representative instead of traversing one family at a time.

## How it works

Each turn the model calls `play(rule, card)`:

- `card` must be a card in its current hand.
- `rule` is a Python boolean expression over `card` and `mainline`.
- The card is accepted or rejected by the hidden rule.
- The hypothesis is checked by deterministic behavioral equivalence.

The episode ends when the hypothesis is correct or after 100 valid plays. Solved episodes receive earlier-is-better reward:

```text
reward = (101 - turns_used) / 100
```

Unsolved episodes receive zero.

## Run

Eleusis requires `verifiers>=0.3.2.dev162,<0.4`. The default remains one agent using the standard chat/tool program. Version 0.8.0 adds configurable teams and updates the SDK; older calibration results retain their original version provenance.

```bash
uv run vf-eval eleusis -m <model>
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
- `--sampling.temperature <x>` — temperature (left at the model default unless set).

## Single-agent and team settings

Use one package and taskset ID, `eleusis`, for all settings:

```bash
# Default: original single-agent prompt, reward and game horizon.
uv run vf-eval eleusis -m <model> -n 1 -r 2

# Cooperative team; any positive team size is supported.
uv run vf-eval eleusis -m <model> --env.num-agents 3 -n 1 -r 2 -c 1

# Local validation with a network-policy-capable runtime.
uv run vf-eval eleusis @ configs/eleusis/cooperative.toml -m <model>
```

The current SDK calls its evaluator `vf-eval`. Prime is the default runtime; Docker is supported for local evaluation. A subprocess runtime cannot enforce the required network policy and is rejected. Install from the repository root with `uv pip install -e ./environments/eleusis`; use `uv run --no-sync vf-eval ...` when the root development environment was installed this way.

`--env.num-agents` defaults to 1. Values greater than one start separate private games with identical hidden rule, starter, hand and draw pile. Agents have separate contexts and tool state. Every multi-agent team has the broadcast tool enabled. The episode stops as soon as a teammate solves. Losing or exhausted agents do not stop their peers.

Cooperative agents can call `send_message(message)` to broadcast arbitrary text to every other teammate. Messages are injected as user messages before the recipient's next model call. They do not play cards, consume a valid play, or submit hypotheses. There are no fixed roles, required messages, agreement phase or forced synchronization during play. Agents that have already ended cannot consume later messages; in-flight model calls are not interrupted by new messages. Each trace records its seat, sent messages and deliveries with send/receive times.

Team rewards are shared binary success: all seats receive 1 if any seat solves, otherwise 0. The original per-seat normalized score is retained as `individual_reward`; `solved`, `turns_used`, and the other original metrics remain individual. `team_solved`, `team_size` and `team_capped_seconds` are repeated on each trace. Consumers must exclude episodes with `ok=false` and aggregate by episode rather than counting each sibling as a separate trial. Team cancellation can leave a seat unsolved with budget remaining; use its stop condition when interpreting `early_abandonment`. Solo reward remains `(max_turns + 1 - turns_used) / max_turns` and its prompt is unchanged.

Team limits are 100 valid plays per seat by default (taskset `max_turns`), 15 invalid actions, `--env.max-calls-per-agent 160` including message-only responses, and `--env.deadline-seconds 1200`. The deadline starts after all harnesses and tool servers are prepared. `--env.setup-timeout 1200` separately bounds team preparation. Agent-level SDK limits can impose tighter limits. Deadline expiration stops pending siblings as an ordinary unsuccessful completion; infrastructure/setup failures propagate, with cancellation-safe sibling cleanup. No whole-episode retries are enabled by default.

Teams run concurrently by default. `--env.max-concurrent-agents` is enforced by the native Verifiers per-episode semaphore during agent setup, interaction turns, and finalization; setting it below the team size changes scheduling and should be recorded in experiments. Evaluation concurrency `-c` multiplies by the team size. Larger teams increase request load and broadcast traffic.

The team controller uses native `Env`, `Agent.interaction`, `Task`, `Toolset` and per-agent traces. Its bundled harness checkpoints the upstream chat program after each response to route messages, retaining tool results and provider wire state rather than reconstructing history from a partial trace. The checkpoint stays inside the configured runtime. The source adaptation is checked against the installed SDK and fails explicitly if its expected structure changes. Custom harness overrides remain supported for solo; teams require this bundled turn-yielding harness. There is no separate multi-agent game engine, rule bank or environment ID.

## Design notes

- **Custom tool:** the `play(rule, card)` tool is the environment. The game state machine must intercept every action, evaluate the hidden rule, update the board, and check the hypothesis by behavioral equivalence — this cannot be reduced to a plain harness task, which is why Eleusis ships a toolset rather than a bare prompt taskset.
- **Isolation:** rollout network access is disabled because the public rule bank contains the answer code. Model-supplied hypotheses run in a reusable child process with CPU, memory, and wall-time limits; an over-limit hypothesis is incorrect and the game continues. A hypothesis must agree with prior verdicts from the rollout and a deterministic probe battery spanning the full game horizon before it is accepted.
- **Prompt:** the system prompt documents the game rules, the expression language, and the scoring contract, and is kept stable because benchmark numbers are measured under it; it intentionally does not teach play strategy.

## Rule dataset format

A rule source must contain unique, non-empty `rule_id` and `code` columns. `code` is a Python boolean expression (or function body) over `card` and `mainline` that returns whether a candidate card is accepted, for example:

```python
return card.color == "red"
```

Optional source control: `--env.taskset.dataset-config <name>` selects a dataset configuration within the rule bank.

### Provider streaming recovery

Standard `vf-eval eleusis` and hosted callers loading `EleusisEnv` automatically retry HTTP 429s and SSE `error.code=429` responses embedded in HTTP 200. The environment wraps each episode's native interception slots and borrows their eval client; it does not modify Verifiers, replace global factories, or change training clients. The native environment retains ownership of servers and client cleanup.

Each relay invocation allows six attempts with exponential backoff (2–32 seconds plus jitter). Failed connections are closed and partial responses discarded before any tool executes. Other errors propagate; the native SDK still rejects incomplete streams. Existing harness SDK retries may repeat an exhausted request, so six is the limit per relay invocation, not per episode. Backoff consumes the existing solve-time budget. Prompts, sampling, game limits and scoring are unchanged. `scripts/eleusis/eval_resilient.py` remains only as an alias for old commands; it no longer patches the SDK.

Solo and cooperative prompts are composed from shared game instructions and explicit mode-specific action/scoring sections. The refactor preserves the previously evaluated prompt text byte-for-byte.

The game tool server is colocated inside each seat's runtime, rather than running as a driver-side subprocess. Package exports load lazily, and rule-checker subprocesses import `eleusis.rules` directly rather than re-importing the tool-server entrypoint and evaluation framework. The existing pipe protocol, behavioral probes and CPU/memory/wall limits are retained. This reduces driver memory; runtime memory still scales with live seats. The native `max_concurrent_agents` gate controls active segments, not the number of resident seats.

The package requires `verifiers>=0.3.2.dev162,<0.4`. The lower bound is the development SDK validated with solo and cooperative Docker evaluations; stable 0.3.1 lacks the native chat-launch API used here and is excluded. Record the resolved SDK version for reproducibility; team checkpoint adaptation fails explicitly if the installed chat program is incompatible.
