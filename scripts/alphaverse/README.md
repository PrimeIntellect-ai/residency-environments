# Alphaverse validation helpers

## Deterministic virtual-clock smoke

From the repository root, install the environment and no-inference probe:

```sh
uv pip install -e ./environments/alphaverse -e ./scripts/alphaverse/clock_probe
uv run eval @ configs/alphaverse/deterministic-clock.local.toml
uv run eval @ configs/alphaverse/deterministic-clock.local.toml --env.agent.harness.delay-seconds 0.25 --run.name clock-delayed
```

Each run evaluates one seeded task with two rollouts. The second inserts 250 ms
before each tool call, while submitting the same order, strategy, and waits.
The config deliberately omits `time_mode`. Each rollout asserts that runtime
delays leave market time unchanged and explicit waits advance it by the requested
amount. It saves `trace.info.clock_probe`, including a hash of the decompressed
canonical event stream and public observations. Those hashes should match across
all four successful rollouts; model calls should remain zero. Use fresh run
names when repeating the check.

### Validation receipt (2026-09-07)

The `clock-wheel-fast` and `clock-wheel-delayed` runs passed with two rollouts
each, seed 7, the balanced `mvp-v1` scenario, and no model calls. All four had
692 identical canonical events and identical public-observation hashes. Market
time stayed at 2 seconds across zero-duration waits and runtime delays, then
advanced to 4 seconds after the explicit 2-second wait.

- Canonical SHA-256: `8883230544794ccae6c560cfe05beb39da84555febd175626a4a8687ed7f1f0c`
- Public observations SHA-256: `a59ad7bac8e66566a0dde3949dd8bcf2cc724848632c2245b43c161ab23c40e3`

This is a deterministic scripted-action smoke check, not a claim that sampled
model actions or arbitrary uploaded strategy programs are deterministic.

## Raw PnL and closeout smoke

The same no-inference probe can leave an open position when its harness exits,
or advance to the market horizon. The environment must liquidate it and export
the terminal artifacts before scoring:

```sh
uv run eval @ configs/alphaverse/deterministic-clock.local.toml --env.agent.harness.exit-mode harness --run.name pnl-early-long
uv run eval @ configs/alphaverse/deterministic-clock.local.toml --env.agent.harness.exit-mode harness --env.agent.harness.side sell --run.name pnl-early-short
uv run eval @ configs/alphaverse/deterministic-clock.local.toml --env.agent.harness.exit-mode horizon --run.name pnl-horizon
```

On 2026-09-22, explicit termination, early long completion, early short
completion, and the horizon each passed one task with two rollouts (eight total).
Each finished flat with reward equal to raw PnL of -4.10, including 0.10 in fees,
with no model calls. Explicit and early completion ended at 4 simulated seconds;
the horizon case ended at 20 seconds. Streamed artifacts were exported in every
case. No unit tests were added.

Two additional early-completion rollouts used `--env.agent.harness.quantity 10`
with a short position and `--env.taskset.task.toolset.artifact-transport inline`.
Both finished flat with raw reward -41.00 (including 1.00 in fees), confirming
that reward is not bounded to the former [-10, 10] range. All ten rollouts
completed without errors, and their Docker runtimes were cleaned up.

## Bounded tools and scored limits

```sh
uv run eval @ configs/alphaverse/bounded-tools.local.toml
uv run eval @ configs/alphaverse/bounded-tools.local.toml --run.name busy-callback --env.agent.harness.strategy-mode busy
uv run eval @ configs/alphaverse/bounded-tools.local.toml --run.name scored-limit --env.episode-time-limit-seconds 8 --env.agent.harness.pause-seconds 60 --env.agent.harness.exit-mode harness
uv run eval @ configs/alphaverse/bounded-tools.local.toml --run.name no-horizon --env.taskset.max-market-time-ns None --env.agent.harness.exit-mode harness
```

The busy strategy deliberately loops forever in `on_start`. Its callback must
fault within the callback deadline, while the market remains usable and the
episode still scores its actual PnL. The pause case verifies that a configured
play-time limit liquidates and scores, rather than being treated as an
infrastructure timeout. These are scripted runtime checks, not model benchmarks.

For the fail-closed hard-deadline path, run:

```sh
uv run eval @ configs/alphaverse/bounded-tools.local.toml --run.name hard-deadline --env.taskset.task.toolset.tool-timeout-seconds 0.25 --env.agent.harness.expect-tool-error true
```

This deliberately interrupts deployment and must produce an explicit server
deadline error and an invalid, unscored rollout—not a trading penalty. Check the
recorded `clock_probe.tool_timings` and verify there are no remaining eval
containers. The `eval` process exit status alone does not establish success:
inspect each saved trace's `ok`, `errors`, and `rewards`.

### Validation receipt (2026-09-30)

Twelve successful local Docker rollouts covered baseline termination, sliced waits,
an infinite-loop strategy callback, an eight-second play-time limit, and a
disabled market horizon, plus horizon closeout (one task, two rollouts per case).
All finished flat and scored raw PnL -4.10. The play-time limit recorded
`outcome="limit_reached"` and `limit_reached="episode_time"`; horizon closeout
recorded `limit_reached="market_time"`. The stuck strategy recorded one strategy fault,
without an additional reward penalty.

The sliced waits yielded 27/28 times with a 20 ms slice. Their canonical event
SHA-256 matched the unsliced baseline exactly:
`8883230544794ccae6c560cfe05beb39da84555febd175626a4a8687ed7f1f0c`.
Two injected 250 ms server deadlines returned explicit tool errors in 0.44/0.50 s
including deployment cleanup. Both were correctly unscored; all eval containers
were released. No per-environment unit tests were added.

A further five scripted rollouts bought ten contracts, held for 300 simulated
seconds, and explicitly liquidated. Seeds 0–4 returned raw PnL
`[1219, -1571, 1359, -531, -51]` (mean 85), with no errors and flat terminal
positions. This is a scoring-mechanics distribution, not a model-performance
benchmark. Reproduce it with:

```sh
uv run eval @ configs/alphaverse/deterministic-clock.local.toml --run.name bounded-pnl-seeds --num-tasks 5 --num-rollouts 1 --env.taskset.seed 0 --env.taskset.max-market-time-ns 1800000000000 --env.agent.harness.wait-duration-ns 300000000000 --env.agent.harness.quantity 10
```

The separate paid-inference natural-completion config is
`configs/alphaverse/natural-completion.local.toml`; scripted checks do not replace
a real-model natural-completion trajectory or model reward-distribution check.
