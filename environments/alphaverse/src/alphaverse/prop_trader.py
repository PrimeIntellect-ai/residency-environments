"""Seed materials for the first session-adaptive proprietary trader."""

PROP_PARTICIPANT_ID = "prop"

PROP_PASSIVE_BASELINE_SOURCE = '''from alphaverse.reference_strategies import EventDrivenAdaptiveMarketMaker


class StrategyImpl(EventDrivenAdaptiveMarketMaker):
    """Small, conservative baseline that remains live between research windows."""

    def __init__(self):
        super().__init__(
            refresh_interval=1_000_000_000,
            initial_refresh_delay=250_000_000,
            quote_quantity=4,
            base_half_spread=3,
            volatility_multiplier=0.75,
            volatility_lookback=12,
            inventory_skew_per_unit=0.2,
            inventory_soft_limit=12,
            minimum_quote_fraction=0.25,
            fill_pressure_per_unit=0.04,
            fill_pressure_half_life=3_000_000_000,
            maximum_fill_skew=3.0,
            markout_horizon=5_000_000_000,
            markout_learning_rate=0.5,
            toxicity_widening_multiplier=1.5,
            toxicity_half_life=15_000_000_000,
            maximum_toxicity_widening=6,
            directional_toxicity=True,
            improvement_reprice_ticks=1,
            retreat_reprice_ticks=1,
            external_book_reference=True,
            replenish_partial_fills=False,
        )
'''


_PROP_COMPETITIVE_BASELINE_TEMPLATE = '''from alphaverse import Side
from alphaverse.strategy import Strategy


class StrategyImpl(Strategy):
    """Small inside-market starter that deliberately produces fill feedback."""

    def __init__(self):
        self.position = 0
        self.best_bid = None
        self.best_ask = None
        self.live = {}
        self.pending = {}
        self.canceling = set()
        self.sequence = 0
        self.quote_quantity = 8
        self.correction_quantity = 12
        self.quote_offset_ticks = 0
        self.inventory_soft_limit = 16
        self.max_abs_position = 64
        self.inventory_retreat_ticks = 1
        self.refresh_interval_ns = 2_000_000_000

    def on_start(self, ctx, event):
        return [ctx.set_timer("quote", fire_at=ctx.now + 250_000_000)]

    def on_levels(self, ctx, event):
        bids = event.payload.get("bids", [])
        asks = event.payload.get("asks", [])
        self.best_bid = int(bids[0]["price"]) if bids else None
        self.best_ask = int(asks[0]["price"]) if asks else None

    def on_execution(self, ctx, event):
        payload = event.payload
        kind = payload.get("event_kind")
        if kind == "order_accepted":
            candidate = self.pending.pop(payload["client_order_id"], None)
            if candidate is not None:
                self.live[payload["order_id"]] = candidate
        elif kind == "order_rejected":
            self.pending.pop(payload.get("client_order_id"), None)
        elif kind in ("cancel_accepted", "cancel_rejected"):
            order_id = payload.get("order_id")
            self.live.pop(order_id, None)
            self.canceling.discard(order_id)
        elif kind == "fill":
            quantity = int(payload["quantity"])
            self.position += quantity if payload["side"] == "buy" else -quantity
            order_id = payload["order_id"]
            if order_id in self.live:
                self.live[order_id]["remaining"] -= quantity
                if self.live[order_id]["remaining"] <= 0:
                    self.live.pop(order_id, None)
                    self.canceling.discard(order_id)
        elif kind == "account":
            self.position = int(payload.get("position", self.position))

    def on_timer(self, ctx, event):
        actions = []
        for order_id in list(self.live):
            if order_id not in self.canceling:
                self.canceling.add(order_id)
                actions.append(ctx.cancel(order_id))

        if (
            not self.pending
            and self.best_bid is not None
            and self.best_ask is not None
            and self.best_ask > self.best_bid
        ):
            bid_price = self.best_bid - self.quote_offset_ticks - (
                self.inventory_retreat_ticks
                if self.position >= self.inventory_soft_limit
                else 0
            )
            ask_price = self.best_ask + self.quote_offset_ticks + (
                self.inventory_retreat_ticks
                if self.position <= -self.inventory_soft_limit
                else 0
            )
            buy_quantity = min(
                self.quote_quantity
                if self.position > -self.inventory_soft_limit
                else self.correction_quantity,
                self.max_abs_position - self.position,
            )
            sell_quantity = min(
                self.quote_quantity
                if self.position < self.inventory_soft_limit
                else self.correction_quantity,
                self.max_abs_position + self.position,
            )
            if self.position >= self.max_abs_position - self.quote_quantity:
                buy_quantity = 0
            if self.position <= -self.max_abs_position + self.quote_quantity:
                sell_quantity = 0
            for label, side, price, quantity in (
                ("bid", Side.BUY, bid_price, buy_quantity),
                ("ask", Side.SELL, ask_price, sell_quantity),
            ):
                if quantity <= 0:
                    continue
                self.sequence += 1
                client_id = f"starter-{label}-{self.sequence}"
                self.pending[client_id] = {
                    "side": label,
                    "price": price,
                    "remaining": quantity,
                }
                actions.append(ctx.submit_limit(client_id, side, price, quantity))

        actions.append(
            ctx.set_timer("quote", fire_at=ctx.now + self.refresh_interval_ns)
        )
        return actions

    def on_risk(self, ctx, event):
        actions = []
        for order_id in list(self.live):
            if order_id not in self.canceling:
                self.canceling.add(order_id)
                actions.append(ctx.cancel(order_id))
        actions.append(ctx.emit_alert("risk", "margin state changed", dict(event.payload)))
        return actions
'''


PROP_KNOB_DEFAULTS = {
    "quote_quantity": 8,
    "correction_quantity": 12,
    "quote_offset_ticks": 0,
    "inventory_soft_limit": 16,
    "max_abs_position": 64,
    "inventory_retreat_ticks": 1,
    "refresh_interval_ns": 2_000_000_000,
}


def _require_knob_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def competitive_prop_source(
    *,
    quote_quantity: int = 8,
    correction_quantity: int = 12,
    quote_offset_ticks: int = 0,
    inventory_soft_limit: int = 16,
    max_abs_position: int = 64,
    inventory_retreat_ticks: int = 1,
    refresh_interval_ns: int = 2_000_000_000,
) -> str:
    """Render the competitive strategy family from bounded, auditable knobs."""

    knobs = {
        "quote_quantity": _require_knob_int("quote_quantity", quote_quantity, minimum=1, maximum=32),
        "correction_quantity": _require_knob_int("correction_quantity", correction_quantity, minimum=1, maximum=32),
        "quote_offset_ticks": _require_knob_int("quote_offset_ticks", quote_offset_ticks, minimum=0, maximum=4),
        "inventory_soft_limit": _require_knob_int("inventory_soft_limit", inventory_soft_limit, minimum=1, maximum=48),
        "max_abs_position": _require_knob_int("max_abs_position", max_abs_position, minimum=16, maximum=120),
        "inventory_retreat_ticks": _require_knob_int(
            "inventory_retreat_ticks",
            inventory_retreat_ticks,
            minimum=0,
            maximum=4,
        ),
        "refresh_interval_ns": _require_knob_int(
            "refresh_interval_ns",
            refresh_interval_ns,
            minimum=250_000_000,
            maximum=10_000_000_000,
        ),
    }
    if knobs["inventory_soft_limit"] >= knobs["max_abs_position"]:
        raise ValueError("inventory_soft_limit must be below max_abs_position")
    if knobs["quote_quantity"] >= knobs["max_abs_position"]:
        raise ValueError("quote_quantity must be below max_abs_position")

    source = _PROP_COMPETITIVE_BASELINE_TEMPLATE
    for name, default in PROP_KNOB_DEFAULTS.items():
        original = f"        self.{name} = {default:_}"
        replacement = f"        self.{name} = {knobs[name]:_}"
        if source.count(original) != 1:
            raise RuntimeError(f"competitive prop template lost knob marker: {name}")
        source = source.replace(original, replacement)
    return source


PROP_COMPETITIVE_BASELINE_SOURCE = competitive_prop_source()


PROP_SEED_PROFILES = ("passive", "competitive")
PROP_FRAMING_PROFILES = ("incumbent", "neutral", "arbitrary")
PROP_CONTROL_SCOPES = ("full_source", "knobs")


def prop_baseline_source(profile: str) -> str:
    """Return the participant seed source for one explicit ablation arm."""

    if profile == "passive":
        return PROP_PASSIVE_BASELINE_SOURCE
    if profile == "competitive":
        return PROP_COMPETITIVE_BASELINE_SOURCE
    raise ValueError(f"unknown prop seed profile: {profile!r}")


_PROP_SYSTEM_PROMPT = """You operate one proprietary trading account in Alphaverse.

Your objective is to maximize your own terminal realized PnL. You receive research
and deployment access only while the exchange is closed between trading sessions.
Your last valid strategy remains active during every open session. At each close,
use newly available public data and your private account history to decide whether
to keep or replace it. You may modify strategy.py and stage a deployment for the
next session. You cannot inspect other participants' private state or source code.

Read README.md, API.md, and ROLE.md. Do not merely describe a strategy change: call
the deployment tool when you want a validated candidate to become active at reopen.
If you do not stage a valid deployment, the previous strategy continues.
"""

_ARBITRARY_STARTER_NOTE = """

The strategy present at the start of the episode is arbitrary scaffolding. It is
not a recommendation, a privileged baseline, or presumed to be close to optimal.
Treat its architecture and every parameter as untrusted hypotheses. Incremental
changes are not preferred: replace the entire strategy when the evidence supports
doing so.
"""

_NEUTRAL_STARTER_NOTE = """

The starter strategy has no privileged status and is not guaranteed to be
optimal. Keeping it, tuning it, or replacing it are equally valid choices.
Decide among them from the evidence you observe.
"""

_NEUTRAL_KNOB_NOTE = """

The starter parameters have no privileged status and are not guaranteed to be
optimal. Keeping or tuning them are equally valid choices. The strategy family
itself is fixed in this arm; decide among its bounded settings from the evidence
you observe.
"""


def prop_system_prompt(framing: str, control_scope: str = "full_source") -> str:
    """Return the role prompt while varying only the starter framing."""

    prompt = _PROP_SYSTEM_PROMPT
    if control_scope == "knobs":
        prompt = prompt.replace(
            "You may modify strategy.py and stage a deployment for the\nnext session.",
            "You may tune only the documented strategy knobs and stage them for "
            "the next session by passing a JSON knob document in the "
            "deploy_strategy tool's source argument. Raw Python strategy-source "
            "deployment is unavailable in this arm.",
        ).replace(
            "call\nthe deployment tool",
            "call\ndeploy_strategy with the JSON knob document",
        )
    elif control_scope != "full_source":
        raise ValueError(f"unknown prop control scope: {control_scope!r}")
    if framing == "incumbent":
        return prompt
    if framing == "neutral":
        note = _NEUTRAL_KNOB_NOTE if control_scope == "knobs" else _NEUTRAL_STARTER_NOTE
        return prompt.rstrip() + note
    if framing == "arbitrary":
        if control_scope == "knobs":
            raise ValueError("arbitrary framing is incompatible with knobs control")
        return prompt.rstrip() + _ARBITRARY_STARTER_NOTE
    raise ValueError(f"unknown prop framing profile: {framing!r}")


_PROP_ROLE_README = """# Prop trader role

You control the `prop` account. `strategy.py` is the source of the {seed_description}
that is already trading. The exchange is closed whenever you are invoked.
Market time, matching, participant callbacks, and margin processing are frozen.

Available exchange tools let you inspect public data, your own account/orders, and
stage strategy source. Direct manual orders, waits, strategy stops, and episode
termination are intentionally unavailable. A successful deployment is validated
now and activated atomically when the next market session opens.

Your workspace persists across every scheduled review in this episode.
"""


def prop_role_readme(
    seed_profile: str,
    framing: str,
    control_scope: str = "full_source",
) -> str:
    """Build agent-visible role documentation for an ablation arm."""

    if seed_profile == "passive":
        description = "small, conservative baseline"
    elif seed_profile == "competitive":
        description = "small inside-market starter"
    else:
        raise ValueError(f"unknown prop seed profile: {seed_profile!r}")
    readme = _PROP_ROLE_README.format(seed_description=description)
    if control_scope == "knobs":
        readme = readme.replace(
            "stage strategy source.",
            "stage only a JSON document containing the bounded parameters in "
            "the deploy_strategy tool's source argument. Raw Python "
            "strategy-source deployment is unavailable.",
        )
    elif control_scope != "full_source":
        raise ValueError(f"unknown prop control scope: {control_scope!r}")
    if framing == "neutral":
        note = _NEUTRAL_KNOB_NOTE if control_scope == "knobs" else _NEUTRAL_STARTER_NOTE
        readme = readme.rstrip() + note
    elif framing == "arbitrary":
        if control_scope == "knobs":
            raise ValueError("arbitrary framing is incompatible with knobs control")
        readme = readme.rstrip() + _ARBITRARY_STARTER_NOTE
    elif framing != "incumbent":
        raise ValueError(f"unknown prop framing profile: {framing!r}")
    return readme


# Stable defaults used by the evaluator roster and research reports.
PROP_BASELINE_SOURCE = prop_baseline_source("passive")
PROP_SYSTEM_PROMPT = prop_system_prompt("incumbent")
PROP_ROLE_README = prop_role_readme("passive", "incumbent")

__all__ = [
    "PROP_BASELINE_SOURCE",
    "PROP_COMPETITIVE_BASELINE_SOURCE",
    "PROP_CONTROL_SCOPES",
    "PROP_FRAMING_PROFILES",
    "PROP_KNOB_DEFAULTS",
    "PROP_PARTICIPANT_ID",
    "PROP_PASSIVE_BASELINE_SOURCE",
    "PROP_ROLE_README",
    "PROP_SEED_PROFILES",
    "PROP_SYSTEM_PROMPT",
    "competitive_prop_source",
    "prop_baseline_source",
    "prop_role_readme",
    "prop_system_prompt",
]
