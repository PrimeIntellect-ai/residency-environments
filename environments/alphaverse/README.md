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
(10,000 by default), clips the result, and subtracts an explicit incomplete-
liquidation penalty. Infrastructure failures invalidate a rollout rather than
becoming trading losses. The exchange also records volume, fills,
orders, rejections, position, drawdown, margin events, strategy deployments,
model usage, and inference cost as diagnostics.

Each task creates one deterministic market from a scenario seed. The default
taskset exposes five balanced-demand seeds with a 1,800-second virtual-market
horizon; command-line configuration can change the seed count, demand profile,
horizon, capital, margin, clock, participant roster, and adaptive-prop settings.

Each task is one seeded example. Multiple rollouts of that task reuse its seed
and initial market; the task count does not introduce different seeds within a
rollout group. The default `time_mode="manual"` uses deterministic virtual time:
inference latency and infrastructure delays do not advance the market, while
explicit waits process scheduled events. The same seed and ordered actions at
the same simulated times reproduce the market trajectory. Wall-clock mode is
available only by explicit configuration and is not latency-independent.

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

A short two-agent deployment/intermission/streaming check is available in
[`configs/alphaverse/isolation-smoke.local.toml`](../../configs/alphaverse/isolation-smoke.local.toml).
It deliberately prompts for mechanical actions and is not an economic benchmark:

```bash
uv run --no-sync eval @ configs/alphaverse/isolation-smoke.local.toml
```

A smaller single-agent Prime deployment/replacement/termination check is in
[`configs/alphaverse/isolation-smoke.prime.toml`](../../configs/alphaverse/isolation-smoke.prime.toml).
It creates short-lived paid micro-VMs and uses a directive mechanics prompt:

```bash
uv run --no-sync eval @ configs/alphaverse/isolation-smoke.prime.toml
```

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

There are three placements, managed by Verifiers:

- **Exchange:** a non-colocated, evaluator-side Toolset (`SubprocessConfig`).
  It owns matching, clearing, latent state, other participants, scoring, and
  private artifacts. Only trusted environment code executes here.
- **Research:** each agent's container or VM receives public documentation,
  its own workspace, and authenticated market tools. General internet access is
  blocked; Verifiers preserves inference and the assigned MCP routes.
- **Trading:** each uploaded deployment gets a fresh runtime through native
  `provision_runtime()` and a persistent `open_process()` channel. The default
  is a Prime micro-VM with `allow=[]`; Docker with `allow=[]` is available for
  local development. Subprocess placement, non-VM Prime placement, and outbound
  network access are rejected. There is no host execution fallback.

Trading runtimes receive only an explicit public-SDK file list and that firm's
source. They receive no evaluator environment variables, shared filesystem,
exchange package, other-firm source, or management credentials. A participant's
initial strategy is supplied as self-contained source, exactly like an update.
The strategy may use ordinary Python and its own local files. Syntax checking
is not the security boundary; the framework's container/VM is.

Only bounded JSON action batches cross back. The exchange validates their schema,
callback ID, action count, ownership, and existing trading/risk rules; worker
output never determines caller identity. Callback deadlines, memory/CPU limits,
and bounded response/diagnostic sizes contain faulty programs. Docker's disk
request is advisory, not a hard per-container quota; use Prime VMs for hostile
multi-tenant evaluations. Isolation relies on Verifiers and the provider, not a
claim that Python or containers are immune to platform vulnerabilities.

Deployments are initialized before replacing the incumbent. Intermission updates
keep their prepared runtime until reopen; a failed readiness check preserves the
incumbent. Stop, replacement, termination, and Toolset teardown release runtimes.
The market remains one implementation: the strategy bridge changes execution
placement, not matching, accounting, market feeds, or the rules of trading.

For an entirely local smoke test (Docker must be running):

```bash
uv run --no-sync eval alphaverse -n 1 -r 2 --env.agent.max-turns 4 \
  --env.agent.runtime.type docker \
  --env.taskset.task.toolset.strategy-runtime.type docker \
  --env.taskset.task.toolset.strategy-runtime.allow '[]'
```

The empty `allow` list blocks trading-runtime egress. Provider credentials remain
with the evaluator; they are not installed in either agent or trading runtimes.

Each delivered strategy callback currently requires a synchronous round trip,
even when it returns no actions. Local Docker and remote Prime use the same
native Runtime API but different transports; wide-area latency can dominate
long runs. Validate placement and callback throughput before scaling the horizon;
the short isolation smoke does not establish six-hour cloud performance.
