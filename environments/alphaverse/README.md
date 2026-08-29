# Alphaverse

Alphaverse is a deterministic limit-order-market environment for evaluating
long-horizon coding agents. The agent researches a synthetic market, writes and
deploys an automated Python strategy, monitors private executions and public
market data, and iterates while virtual market time continues to advance.

The package contains only the executable environment: exchange and clearing
logic, participant ecology, player strategy runtime, Verifiers v1 task and tool
server, agent workspace, scoring, artifacts, and optional adaptive-participant
orchestration. Research notebooks, dashboards, documentation sites, historical
rollouts, and presentation assets are intentionally excluded.

## Objective

The primary objective metric is terminal realized PnL after transaction fees
and terminal liquidation. The scalar reward divides PnL by a configurable scale
(10,000 by default), clips the result, and subtracts explicit incomplete-
liquidation or rollout-error penalties. The exchange also records volume, fills,
orders, rejections, position, drawdown, margin events, strategy deployments,
model usage, and inference cost as diagnostics.

Each task creates one deterministic market from a scenario seed. The default
taskset exposes five balanced-demand seeds with a 1,800-second virtual-market
horizon; command-line configuration can change the seed count, demand profile,
horizon, capital, margin, clock, participant roster, and adaptive-prop settings.

## Mechanics

- One perpetual futures-like instrument with integer ticks and price-time FIFO.
- Anonymous market-by-order changes, top-level snapshots, trades, and private
  execution/account/risk messages.
- Market makers, noise and mandate traders, momentum participants, informed
  execution styles, and latent demand/anchor processes.
- Initial and maintenance margin, reduce-only transitions, forced liquidation,
  uniform per-contract fees, and terminal flattening.
- Event-driven uploaded strategies with timers, alerts, risk callbacks, bounded
  actions, callback deadlines, and deterministic random state.
- Optional scheduled intermissions in which a second agent may revise a
  proprietary participant before the next market session.

Task setup installs only the public files in `src/alphaverse/agent_workspace/`,
so stock coding harnesses can play without an Alphaverse-specific provisioning
step. Evaluated agents cannot inspect exchange, participant, evaluator, or
latent-state implementation. Agent runtimes use framework-only networking;
Codex web search and Claude Code web tools are disabled in the optional native
harness adapters. The bundled adapters additionally configure direct-to-file
market capture and terminal artifact streaming after Toolset URLs exist.

## Install and validate

From the repository root:

```bash
uv pip install -e ./environments/alphaverse
uv run ruff check ./environments/alphaverse
uv run ruff format --check ./environments/alphaverse
uv build ./environments/alphaverse
```

Run the standard smoke evaluation with the bundled coding harness:

```bash
uv run --no-sync eval alphaverse -n 1 -r 2 --env.agent.max-turns 4
```

The package also exposes `alphaverse_codex_harness`,
`alphaverse_claude_code_harness`, and `alphaverse_adaptive_env` for explicit
experiments. The default `alphaverse` taskset remains a single-player evaluation
and does not silently add an adaptive opponent.

## Agent interface

The complete agent-visible rules and API are bundled as:

- `src/alphaverse/agent_workspace/README.md`
- `src/alphaverse/agent_workspace/API.md`
- `src/alphaverse/agent_workspace/market_capture.py`

The market-capture helper downloads already-delivered public MBO or level-feed
records to NDJSON in the agent workspace. It never includes private messages or
hidden participant labels. Direct-to-file capture requires one of the bundled
Alphaverse harness adapters; stock harnesses can use the equivalent bounded
`capture_market_data` tool pages.

## Security boundary

General agent internet access is blocked while Verifiers preserves model
inference and Alphaverse MCP routes. Uploaded strategy source is restricted to
the documented SDK and a small deterministic standard-library allowlist, and
the child worker denies ordinary file, socket, process, reflection, dynamic-code,
and private-package access.

The Python strategy policy is defense in depth, not a kernel-enforced sandbox
for hostile code. Deployments accepting arbitrary untrusted human source should
add a dedicated restricted container or Prime sandbox around each strategy
worker.
