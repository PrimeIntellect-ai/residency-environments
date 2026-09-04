# Alphaverse player workspace

You are operating an automated trading account in a simulated, continuously
running exchange. Your objective is to maximize terminal realized profit and
loss (PnL) while managing execution risk.

This directory is your private scratch workspace. It intentionally contains
only the public player materials:

- `README.md`: game objective and evaluation constraints;
- `API.md`: authoritative tool, event, and strategy interface reference;
- `market_capture.py`: raw public-feed downloader for local data analysis.

You cannot inspect the exchange implementation, participant code, latent
signals, evaluator, or other hidden state. Do not assume their behavior. Infer
market structure from the public feed and your own executions.

The runtime has no general internet access. Only the framework channels needed
for model inference and the documented Alphaverse tools remain reachable. Do
not depend on external websites, package registries, repositories, or services.

## Objective and evaluation

Your primary reward is terminal realized PnL:

```text
terminal cash - starting cash
```

Ending the session cancels your live orders and aggressively liquidates any
remaining position through ordinary book liquidity. Fees and liquidation
slippage therefore count. You may flatten more carefully before terminating.

A session that reaches its disclosed market-time horizon is finalized
automatically. Ending the harness before explicit termination or that horizon is
an incomplete rollout. Open positions or failed liquidation are penalized.

## Time

All market timestamps and wait durations are integer nanoseconds:

```text
1 second = 1_000_000_000 ns
```

Model inference takes real time and the market advances while you think. The
`wait` tool deliberately advances virtual time without requiring model tokens.
All other participants continue running during both kinds of elapsed time. The
task prompt gives the exact market-time horizon for the episode. It applies to
research and periods when no automated strategy is deployed. Reaching it ends
the episode, cancels live orders, and attempts ordinary-book liquidation just
like explicit termination. You may use bounded `wait` intervals to let a
deployed strategy run without emitting model tokens.

Some episodes use scheduled market sessions. In those episodes, wait responses
include `market_session`. At an `intermission` the entire simulation clock is
frozen and inference is not charged as market time. You receive a dedicated
research/deployment turn before the host reopens the next session.

## Important constraints

- Prices are integer ticks and quantities are integer contracts.
- Orders are good-until-cancelled limit orders; aggressive limit prices can
  cross the book.
- Order submission and cancellation are asynchronous. Confirm outcomes through
  private events.
- The delivered level snapshot can lag the exchange and is not an
  acknowledgement of your private order.
- The authoritative account endpoint includes cash, position, cumulative fees,
  and the exchange's current margin mark and state.
- Initial and maintenance margin are fixed per contract and shown by
  `product_terms()`. New orders that would exceed available initial margin are
  rejected. Working orders on the relevant side count toward the projected
  position used by this pre-trade check.
- If marked equity falls below maintenance margin, the exchange cancels your
  live orders and makes the account reduce-only. If the deficit remains through
  the documented grace period, it aggressively reduces the position through
  ordinary book liquidity until initial margin is restored, if liquidity is
  available.
- Margin state changes and order rejections are delivered to strategy
  callbacks. A strategy may use `ctx.emit_alert(...)` when it wants the next
  default `wait` call to return early to the model.
