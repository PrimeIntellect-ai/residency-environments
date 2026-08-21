# Alphaverse agent API

The available tools are prefixed by the Alphaverse MCP server in the harness.
Tool descriptions are authoritative if the displayed prefix differs from the
short names below.

## Research and account tools

### `product_terms()`

Returns product ID, tick and multiplier fields, fixed-point cash scale, and the
uniform transaction fee per filled contract per side. It also returns the
exchange-enforced `risk_limits`, fixed per-contract `margin` terms, and the account's `technology` settings,
including data, decision, and order-entry latency. `simulation_limits` states
the deterministic work budget: if a slice exceeds its scheduled-event limit
while focal automation is active, that automation is faulted and removed while
the episode continues.

### `market_snapshot(depth=10)`

Returns the most recently delivered level-book snapshot:

```json
{
  "market_time": 1200000000,
  "available_at": 1200000000,
  "source_event_seq": 42,
  "bids": [{"price": 9998, "quantity": 12, "order_count": 3}],
  "asks": [{"price": 10001, "quantity": 9, "order_count": 2}]
}
```

This is delivered market data, not necessarily current authoritative exchange
state. Empty sides are possible.

### `events(after_cursor=null, limit=100)`

Returns public market data and your private messages:

```json
{
  "events": [
    {
      "cursor": 7,
      "event_id": "E42",
      "kind": "levels",
      "exchange_time": 1200000000,
      "available_at": 1200000000,
      "source_event_seq": 42,
      "payload": {"event_kind": "levels"}
    }
  ],
  "next_after_cursor": 7
}
```

`limit` must be between 1 and 200. Use `next_after_cursor` to paginate rather
than requesting an unbounded conversational response.

Omitting `after_cursor` continues after the toolset's current cursor. Important
`payload.event_kind` values include:

- `trade`: public price, quantity, aggressor side, and anonymous order IDs;
- `mbo_change`: public anonymous add/reduce/delete when MBO-entitled;
- `levels`: top levels, `through_event_seq`, and `event_end=true`;
- `order_accepted` / `order_rejected`: private order result;
- `cancel_accepted` / `cancel_rejected`: private cancel result;
- `fill`: private side, price, quantity, liquidity role, and fee;
- `account`: private cash, position, and cumulative fees;
- `risk`: private margin state transition and deadline;
- `alert`: durable private notification explicitly emitted by your strategy;
- `session`: private lifecycle state.

Private execution payloads have these exact strategy-relevant fields (in
addition to `event_kind`, timestamps, and envelope metadata):

```text
order_accepted: client_order_id, order_id, side, price, quantity, liquidation, liquidation_reason
order_rejected: client_order_id, reason, and margin diagnostics when applicable
cancel_accepted: order_id
cancel_rejected: order_id, reason
fill: trade_id, order_id, side, price, quantity, liquidity_role, fee, fee_subunits
account: cash, cash_subunits, fees_paid, fees_paid_subunits, position
risk: state, previous_state, mark, mark_source, mark_time, equity,
      initial_requirement, maintenance_requirement, reduce_only,
      liquidation_deadline
alert: code, message, data
```

A `fill` does **not** contain `client_order_id` or `remaining_quantity`. Track
the accepted order's original quantity by `order_id` and subtract each fill
quantity. A rejected cancel means the referenced order is no longer
cancellable. Reissuing that cancel directly from its rejection callback occurs
at the same market time and can exceed the strategy's simulation-work budget.

### Raw market-data capture

Bulk public data should be downloaded directly to your workspace rather than
returned through conversational tool results:

```bash
python market_capture.py spec --feed mbo
python market_capture.py spec --feed levels --output levels.schema.json

python market_capture.py capture \
  --feed mbo \
  --after-cursor 0 \
  --output market.mbo.ndjson
```

The direct-to-file helper is configured by the bundled Alphaverse harnesses. If
the current harness does not provide that private transport file, request the
same bounded raw pages with `capture_market_data` and write the returned records
to disk yourself.

The capture command reads already delivered packets; use `wait` separately when
you want more market time to pass. It prints only a small download summary. The
raw records remain in the output file for analysis with your own code. A capture
freezes its terminal cursor on the first page and transfers bounded pages, so
long histories remain coherent without loading the full tape into memory.

Two feeds are available:

- `mbo`: raw `mbo_change` packets plus public `trade` packets;
- `levels`: raw `levels` packets plus public `trade` packets.

Captures never include private order, fill, account, risk, or session messages.
They do not aggregate, resample, calculate features, reconstruct a book, or
otherwise process the selected packets. The level packet's depth is determined
by the exchange feed.

Capture files are UTF-8 NDJSON with exactly one JSON object per line. The
authoritative record schema is returned by `market_capture_spec(feed)` or the
`market_capture.py spec` command. Each record uses the same envelope as
`events`: `cursor`, `event_id`, `kind`, `exchange_time`, `available_at`,
`source_event_seq`, and `payload`.

`after_cursor` is exclusive and belongs to the complete delivered event stream,
so gaps between captured cursors are expected when private or other feed
packets were filtered out. The command summary returns `next_after_cursor`.
Pass that value to a later capture to download only newly delivered packets:

```bash
python market_capture.py capture \
  --feed mbo \
  --after-cursor NEXT_CURSOR \
  --append \
  --output market.mbo.ndjson
```

`--append` appends only the newly selected records. Without it, the destination
is atomically replaced.

### `account()`

Returns authoritative `starting_cash`, `cash`, `position`, cumulative
`fees_paid`, exact subunit fields, session state, and current `margin` state.
The margin object includes the persisted external mark, marked equity, initial
and maintenance requirements, available initial margin, reduce-only status,
and any liquidation deadline.

The exchange's risk mark is the latest midpoint for which both bid and ask have
orders from participants other than you. It persists across temporarily
one-sided or empty books, so your own quote cannot improve your margin state.

With the default terms, initial margin is 5,000 cash units per contract and
maintenance margin is 4,000. A new order is rejected with
`insufficient_initial_margin` when marked equity would not cover the projected
position after that order and same-side working quantity. If marked equity
falls below maintenance, a `risk` transition to `margin_call` is delivered,
all working orders are cancelled, and the account becomes reduce-only. If it
remains below maintenance at the reported deadline, the exchange crosses
ordinary book liquidity for the minimum available reduction expected to
restore initial margin. Forced fills pay normal fees and slippage. If there is
not enough liquidity, liquidation resumes when new liquidity appears.

### `open_orders()`

Returns your authoritative live orders with exchange order ID, client order ID,
side, price, remaining quantity, and FIFO priority.

## Direct order tools

### `submit_limit_order(client_order_id, side, price, quantity)`

Queues a `buy` or `sell` limit order. The response only confirms queueing for
delivery. Exchange acceptance or rejection arrives asynchronously in `events`.

### `cancel_order(order_id)`

Queues cancellation using the exchange order ID returned by a private
`order_accepted` event or `open_orders`.

## Automated strategy tools

### `deploy_strategy(source, entrypoint="strategy:StrategyImpl")`

Validates and launches the complete Python source string in a dedicated child
process. A successful response includes an immutable SHA-256 version ID.
Redeploying preserves cash and position, stops the previous strategy, cancels
its orders, and starts the new version.

During a scheduled market intermission, validation still happens immediately but
the deployment is staged. The response includes `staged=true`; the previous
strategy remains production-active until all agents finish the intermission and
the host atomically reopens the market.

The public strategy imports are:

```python
from alphaverse import Side
from alphaverse.strategy import Strategy, StrategyContext, InputEnvelope
```

Uploaded modules may use the public SDK names shown above and named imports from
a small deterministic standard-library allowlist (`collections`, `decimal`,
`enum`, `fractions`, `functools`, `heapq`, `itertools`, `math`, `statistics`,
and `typing`). Module imports, filesystem/network/process APIs, dynamic code,
reflection helpers, dunder access, and private Alphaverse modules are rejected.
Package third-party dependencies into your outer research workflow, not the
deployed strategy.

Define `StrategyImpl(Strategy)`. Available callbacks are:

```python
on_start(ctx, event)
on_market(ctx, event)       # public trade or MBO event
on_levels(ctx, event)       # coherent derived level snapshot
on_execution(ctx, event)    # your private order/fill/account/session event
on_timer(ctx, event)
on_risk(ctx, event)
on_stop(ctx, event)
```

Each callback returns an iterable of actions or `None`. Construct actions with:

```python
ctx.submit_limit(client_order_id, Side.BUY, price, quantity)
ctx.submit_limit(client_order_id, Side.SELL, price, quantity)
ctx.cancel(exchange_order_id)
ctx.set_timer(timer_id, fire_at=absolute_market_time_ns)
ctx.cancel_timer(timer_id)
ctx.log(message, fields={...})
ctx.emit_alert(code, message, data={...})
ctx.stop(reason)
```

`ctx.now`, `ctx.exchange_time`, and `ctx.source_event_seq` update before every
callback. `event.payload` contains the fields described under `events`.
Strategy state stored on `self` persists between callbacks for that deployment.
Callback exceptions and timeouts stop the automation and appear in
`strategy_status`.

`ctx.emit_alert` writes a durable private alert for the model; it does not feed
the alert back into `on_alert`. Codes are at most 64 characters, messages 512
characters, and `data` must be a small JSON-compatible mapping. The strategy
chooses when an event deserves an alert: exchange risk events and rejections do
not automatically wake the model.

### `strategy_status()`

Returns whether automation is active, its current version ID, fault, and stop
reason.

### `stop_strategy()`

Stops automation and cancels its live orders without flattening the account.

## Time and termination

### `wait(duration_ns=null, until_ns=null, wake_on_alert=true)`

Provide exactly one time argument. All deployed strategies and participants run
during the interval. The call is synchronous. By default it returns at the next
bounded simulation checkpoint after a strategy-authored alert. Its response
contains `market_time`, the original `requested_until`, `woke_on_alert`, and the
alerts that caused the wake. Returned alerts are acknowledged, so a later wait
does not wake for the same alert. Set `wake_on_alert=false` to run through
alerts. For long episodes, use checkpoint intervals no larger than one market
hour so the request completes within the tool timeout and unsafe behavior can
be stopped.

Scheduled episodes clamp waits exactly at the next session boundary. The response
then reports `market_session.state="intermission"`; additional waits cannot advance
time until the host reopens the next session.

### `terminate_session()`

Stops play, cancels live orders, and attempts aggressive liquidation. This is
irreversible and should be your final tool call.
