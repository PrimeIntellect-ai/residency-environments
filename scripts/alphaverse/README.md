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
