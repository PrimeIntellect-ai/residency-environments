"""Integrated deterministic exchange kernel for the first Alphaverse vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from alphaverse.clearing import AccountSnapshot, ClearingLedger, midpoint
from alphaverse.clock import VirtualClock
from alphaverse.event_log import EventLog
from alphaverse.models import (
    BookChange,
    CancelOrder,
    Event,
    EventKind,
    Fill,
    MatchResult,
    NewOrder,
    Product,
    Side,
)
from alphaverse.order_book import BookLevels, CancelResult, OrderBook
from alphaverse.profiles import MarginConfig


class SessionState(str, Enum):
    ACTIVE = "active"
    LIQUIDATING = "liquidating"
    TERMINATED = "terminated"


class MarginState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    MARGIN_CALL = "margin_call"


@dataclass(frozen=True, slots=True)
class CommandResult:
    accepted: bool
    events: tuple[Event, ...]
    order_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TerminationResult:
    state: SessionState
    remaining_position: int
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class AccountMetrics:
    account: AccountSnapshot
    best_bid: int | None
    best_ask: int | None
    mark: Fraction | None
    marked_equity: Fraction | None
    pnl: Fraction | None


@dataclass(frozen=True, slots=True)
class MarginMetrics:
    """Current margin state for an account with margin enabled."""

    participant_id: str
    state: MarginState
    mark: Fraction | None
    mark_source: str | None
    mark_time: int | None
    marked_equity: Fraction | None
    initial_requirement: int
    maintenance_requirement: int
    available_margin: Fraction | None
    reduce_only: bool
    liquidation_deadline: int | None


class Exchange:
    """One-product exchange, clearing ledger, clock, and canonical event log."""

    def __init__(
        self,
        product: Product | None = None,
        *,
        level_depth: int = 10,
        start_time: int = 0,
    ) -> None:
        if level_depth < 0:
            raise ValueError("level_depth must be non-negative")
        self.product = product or Product()
        self.level_depth = level_depth
        self.clock: VirtualClock[object] = VirtualClock(start_time)
        self.book = OrderBook(self.product.product_id)
        self.clearing = ClearingLedger(
            product_id=self.product.product_id,
            contract_multiplier=self.product.contract_multiplier,
            cash_subunits_per_tick=self.product.cash_subunits_per_tick,
            transaction_fee_per_contract_subunits=(self.product.transaction_fee_per_contract_subunits),
        )
        self.event_log = EventLog()
        self._participants: set[str] = set()
        self._states: dict[str, SessionState] = {}
        self._margin_configs: dict[str, MarginConfig | None] = {}
        self._margin_states: dict[str, MarginState] = {}
        self._margin_call_deadlines: dict[str, int | None] = {}
        self._last_valid_mark: Fraction | None = None
        self._last_valid_margin_marks: dict[str, tuple[Fraction, int]] = {}
        self._processing_margin_calls = False
        self._liquidation_queue: list[str] = []
        self._next_match_event_number = 1

    def register_account(
        self,
        participant_id: str,
        *,
        starting_cash: int,
        margin: MarginConfig | None = None,
    ) -> None:
        self.clearing.register_account(participant_id, starting_cash=starting_cash)
        self._participants.add(participant_id)
        self._states[participant_id] = SessionState.ACTIVE
        self._margin_configs[participant_id] = margin
        self._margin_states[participant_id] = MarginState.NORMAL
        self._margin_call_deadlines[participant_id] = None

    def state(self, participant_id: str) -> SessionState:
        self._require_participant(participant_id)
        return self._states[participant_id]

    def account_metrics(self, participant_id: str) -> AccountMetrics:
        account = self.clearing.snapshot(participant_id)
        best_bid = self.book.best_bid
        best_ask = self.book.best_ask
        self._capture_valid_mark()
        mark = self._last_valid_mark
        if mark is None:
            return AccountMetrics(account, best_bid, best_ask, None, None, None)
        equity = self.clearing.marked_equity_at_mark(participant_id, mark=mark)
        return AccountMetrics(
            account=account,
            best_bid=best_bid,
            best_ask=best_ask,
            mark=mark,
            marked_equity=equity,
            pnl=equity - account.starting_cash,
        )

    def margin_metrics(self, participant_id: str) -> MarginMetrics | None:
        """Return account margin state, or ``None`` when margin is disabled."""

        self._require_participant(participant_id)
        config = self._margin_configs[participant_id]
        if config is None:
            return None
        account = self.clearing.snapshot(participant_id)
        risk_mark = self._last_valid_margin_marks.get(participant_id)
        mark = None if risk_mark is None else risk_mark[0]
        equity = None if mark is None else self.clearing.marked_equity_at_mark(participant_id, mark=mark)
        position = abs(account.position)
        initial = position * config.initial_margin_per_contract
        maintenance = position * config.maintenance_margin_per_contract
        return MarginMetrics(
            participant_id=participant_id,
            state=self._margin_states[participant_id],
            mark=mark,
            mark_source=(None if mark is None else "external_two_sided_bbo"),
            mark_time=(None if risk_mark is None else risk_mark[1]),
            marked_equity=equity,
            initial_requirement=initial,
            maintenance_requirement=maintenance,
            available_margin=(None if equity is None else equity - initial),
            reduce_only=self._margin_states[participant_id] is MarginState.MARGIN_CALL,
            liquidation_deadline=self._margin_call_deadlines[participant_id],
        )

    def submit_order(
        self,
        command: NewOrder,
        *,
        market_time: int | None = None,
    ) -> CommandResult:
        self._set_market_time(market_time)
        before = len(self.event_log)
        match_event_id = self._next_match_event_id()

        reason = self._order_rejection_reason(command)
        if reason is not None:
            diagnostics = self._margin_rejection_diagnostics(command, reason)
            self._append(
                match_event_id,
                EventKind.ORDER_REJECTED,
                {
                    "participant_id": command.participant_id,
                    "client_order_id": command.client_order_id,
                    "reason": reason,
                    **diagnostics,
                },
            )
            return CommandResult(False, self._events_from(before), reason=reason)

        result = self.book.submit(command)
        self._append_order_accepted(match_event_id, command, result.order_id)
        self._record_match_result(match_event_id, result)
        self._process_due_margin_calls()

        # Newly resting liquidity can allow an already terminating participant
        # to continue liquidation. This happens after the external command's
        # complete public event, preserving a deterministic causal boundary.
        self._resume_liquidations()
        return CommandResult(
            True,
            self._events_from(before),
            order_id=result.order_id,
        )

    def reject_order(
        self,
        command: NewOrder,
        reason: str,
        *,
        market_time: int | None = None,
        diagnostics: dict[str, int | float | str | bool | None] | None = None,
    ) -> CommandResult:
        """Record an ordinary private rejection imposed above the book layer."""

        if not reason:
            raise ValueError("reason must not be empty")
        self._set_market_time(market_time)
        before = len(self.event_log)
        self._append(
            self._next_match_event_id(),
            EventKind.ORDER_REJECTED,
            {
                "participant_id": command.participant_id,
                "client_order_id": command.client_order_id,
                "reason": reason,
                **(diagnostics or {}),
            },
        )
        return CommandResult(False, self._events_from(before), reason=reason)

    def cancel_order(
        self,
        command: CancelOrder,
        *,
        market_time: int | None = None,
    ) -> CommandResult:
        self._set_market_time(market_time)
        before = len(self.event_log)
        match_event_id = self._next_match_event_id()

        if command.participant_id not in self._participants:
            reason = "unknown_participant"
            self._append_cancel_rejected(match_event_id, command, reason)
            return CommandResult(False, self._events_from(before), reason=reason)
        if self._states[command.participant_id] is not SessionState.ACTIVE:
            reason = "session_not_active"
            self._append_cancel_rejected(match_event_id, command, reason)
            return CommandResult(False, self._events_from(before), reason=reason)
        if command.product_id != self.product.product_id:
            reason = "unknown_product"
            self._append_cancel_rejected(match_event_id, command, reason)
            return CommandResult(False, self._events_from(before), reason=reason)

        result = self.book.cancel(command)
        if not result.cancelled:
            self._append_cancel_rejected(match_event_id, command, result.reason or "rejected")
            return CommandResult(
                False,
                self._events_from(before),
                reason=result.reason,
            )

        self._append(
            match_event_id,
            EventKind.CANCEL_ACCEPTED,
            {
                "participant_id": command.participant_id,
                "order_id": command.order_id,
            },
        )
        self._record_book_changes(match_event_id, result.book_changes)
        self._publish_levels(match_event_id)
        self._evaluate_margin_states()
        self._process_due_margin_calls()
        return CommandResult(True, self._events_from(before), order_id=command.order_id)

    def process_margin(self, *, market_time: int | None = None) -> tuple[Event, ...]:
        """Advance margin deadlines and force any due deterministic liquidation."""

        self._set_market_time(market_time)
        before = len(self.event_log)
        self._capture_valid_mark()
        self._evaluate_margin_states()
        self._process_due_margin_calls()
        return self._events_from(before)

    def request_termination(
        self,
        participant_id: str,
        *,
        market_time: int | None = None,
    ) -> TerminationResult:
        self._set_market_time(market_time)
        self._require_participant(participant_id)
        before = len(self.event_log)

        if self._states[participant_id] is SessionState.TERMINATED:
            return TerminationResult(
                SessionState.TERMINATED,
                0,
                (),
            )

        if self._states[participant_id] is SessionState.ACTIVE:
            self._states[participant_id] = SessionState.LIQUIDATING
            self._liquidation_queue.append(participant_id)
            match_event_id = self._next_match_event_id()
            self._append(
                match_event_id,
                EventKind.SESSION,
                {"participant_id": participant_id, "state": "liquidation_started"},
            )
            changes = self._cancel_participant_orders(participant_id)
            if changes:
                self._record_book_changes(match_event_id, changes)
                self._publish_levels(match_event_id)

        self._liquidate_once(participant_id)
        account = self.clearing.snapshot(participant_id)
        return TerminationResult(
            self._states[participant_id],
            account.position,
            self._events_from(before),
        )

    def _record_match_result(self, match_event_id: str, result: MatchResult) -> None:
        touched_accounts: set[str] = set()
        for fill in result.fills:
            self.clearing.apply_fill(fill)
            touched_accounts.add(fill.buyer_participant_id)
            touched_accounts.add(fill.seller_participant_id)
            self._record_trade(match_event_id, fill)
            self._record_private_fills(match_event_id, fill)

        self._record_book_changes(match_event_id, result.book_changes)

        for participant_id in sorted(touched_accounts):
            account = self.clearing.snapshot(participant_id)
            self._append(
                match_event_id,
                EventKind.ACCOUNT,
                {
                    "participant_id": participant_id,
                    "cash": account.cash,
                    "cash_subunits": account.cash_subunits,
                    "fees_paid": account.fees_paid,
                    "fees_paid_subunits": account.fees_paid_subunits,
                    "position": account.position,
                },
            )

        if result.book_changes:
            self._publish_levels(match_event_id)
        self._capture_valid_mark()
        self._evaluate_margin_states()

    def _record_trade(self, match_event_id: str, fill: Fill) -> None:
        self._append(
            match_event_id,
            EventKind.TRADE,
            {
                "trade_id": fill.trade_id,
                "price": fill.price,
                "quantity": fill.quantity,
                "aggressor_side": fill.aggressor_side.name.lower(),
                "maker_order_id": fill.maker_order_id,
                "taker_order_id": fill.taker_order_id,
            },
        )

    def _record_private_fills(self, match_event_id: str, fill: Fill) -> None:
        maker_side = fill.aggressor_side.opposite
        fee_subunits = self.clearing.fee_subunits(fill.quantity)
        fee = Fraction(
            fee_subunits,
            self.product.cash_subunits_per_tick,
        )
        for participant_id, order_id, side, role in (
            (
                fill.maker_participant_id,
                fill.maker_order_id,
                maker_side,
                "maker",
            ),
            (
                fill.taker_participant_id,
                fill.taker_order_id,
                fill.aggressor_side,
                "taker",
            ),
        ):
            self._append(
                match_event_id,
                EventKind.FILL,
                {
                    "participant_id": participant_id,
                    "trade_id": fill.trade_id,
                    "order_id": order_id,
                    "side": side.name.lower(),
                    "price": fill.price,
                    "quantity": fill.quantity,
                    "liquidity_role": role,
                    "fee": _json_number(fee),
                    "fee_subunits": fee_subunits,
                },
            )

    def _record_book_changes(
        self,
        match_event_id: str,
        changes: tuple[BookChange, ...],
    ) -> None:
        for change in changes:
            self._append(
                match_event_id,
                EventKind.MBO_CHANGE,
                {
                    "action": change.kind.value,
                    "order_id": change.order_id,
                    "side": change.side.name.lower(),
                    "price": change.price,
                    "remaining_quantity": change.remaining_quantity,
                    "priority": change.priority,
                    "event_end": False,
                },
            )

    def _publish_levels(self, match_event_id: str) -> None:
        levels = self.book.top_k(self.level_depth)
        through_event_seq = self.event_log.last_sequence
        self._append(
            match_event_id,
            EventKind.LEVELS,
            {
                "depth": self.level_depth,
                "through_event_seq": through_event_seq,
                "bids": self._levels_data(levels, Side.BUY),
                "asks": self._levels_data(levels, Side.SELL),
                "event_end": True,
            },
        )

    @staticmethod
    def _levels_data(levels: BookLevels, side: Side) -> list[dict[str, int]]:
        selected = levels.bids if side is Side.BUY else levels.asks
        return [
            {
                "price": level.price,
                "quantity": level.total_quantity,
                "order_count": level.order_count,
            }
            for level in selected
        ]

    def _cancel_participant_orders(self, participant_id: str) -> tuple[BookChange, ...]:
        changes: list[BookChange] = []
        owned_order_ids = [order.order_id for order in self.book.orders_for_participant(participant_id)]
        for order_id in owned_order_ids:
            result: CancelResult = self.book.cancel(CancelOrder(participant_id, order_id, self.product.product_id))
            changes.extend(result.book_changes)
        return tuple(changes)

    def _liquidate_once(self, participant_id: str) -> None:
        if self._states[participant_id] is not SessionState.LIQUIDATING:
            return

        account = self.clearing.snapshot(participant_id)
        if account.position == 0:
            self._complete_termination(participant_id)
            return

        snapshot = self.book.snapshot()
        if account.position > 0:
            opposite = snapshot.bids
            side = Side.SELL
        else:
            opposite = snapshot.asks
            side = Side.BUY

        available_quantity = sum(order.remaining_quantity for order in opposite)
        liquidation_quantity = min(abs(account.position), available_quantity)
        if liquidation_quantity == 0:
            self._append_liquidation_waiting(participant_id, account.position)
            return

        # Snapshot sides are best-to-worst; the final price crosses every unit
        # included in this bounded liquidation slice without leaving a residual.
        limit_price = opposite[-1].price
        match_event_id = self._next_match_event_id()
        command = NewOrder(
            participant_id=participant_id,
            client_order_id=f"__liquidation_{match_event_id}",
            side=side,
            price=limit_price,
            quantity=liquidation_quantity,
            product_id=self.product.product_id,
        )
        result = self.book.submit(command)
        self._append_order_accepted(
            match_event_id,
            command,
            result.order_id,
            liquidation_reason="terminal",
        )
        self._record_match_result(match_event_id, result)

        remaining = self.clearing.snapshot(participant_id).position
        if remaining == 0:
            self._complete_termination(participant_id)
        else:
            self._append_liquidation_waiting(participant_id, remaining)

    def _resume_liquidations(self) -> None:
        for participant_id in tuple(self._liquidation_queue):
            self._liquidate_once(participant_id)

    def _complete_termination(self, participant_id: str) -> None:
        self._states[participant_id] = SessionState.TERMINATED
        if participant_id in self._liquidation_queue:
            self._liquidation_queue.remove(participant_id)
        match_event_id = self._next_match_event_id()
        account = self.clearing.snapshot(participant_id)
        self._append(
            match_event_id,
            EventKind.SESSION,
            {
                "participant_id": participant_id,
                "state": "terminated",
                "terminal_cash": account.cash,
                "terminal_cash_subunits": account.cash_subunits,
                "fees_paid": account.fees_paid,
                "fees_paid_subunits": account.fees_paid_subunits,
                "position": account.position,
            },
        )

    def _append_liquidation_waiting(self, participant_id: str, position: int) -> None:
        self._append(
            self._next_match_event_id(),
            EventKind.SESSION,
            {
                "participant_id": participant_id,
                "state": "liquidation_waiting",
                "position": position,
            },
        )

    def _append_order_accepted(
        self,
        match_event_id: str,
        command: NewOrder,
        order_id: str,
        *,
        liquidation_reason: str | None = None,
    ) -> None:
        self._append(
            match_event_id,
            EventKind.ORDER_ACCEPTED,
            {
                "participant_id": command.participant_id,
                "client_order_id": command.client_order_id,
                "order_id": order_id,
                "side": command.side.name.lower(),
                "price": command.price,
                "quantity": command.quantity,
                "liquidation": liquidation_reason is not None,
                "liquidation_reason": liquidation_reason,
            },
        )

    def _append_cancel_rejected(
        self,
        match_event_id: str,
        command: CancelOrder,
        reason: str,
    ) -> None:
        self._append(
            match_event_id,
            EventKind.CANCEL_REJECTED,
            {
                "participant_id": command.participant_id,
                "order_id": command.order_id,
                "reason": reason,
            },
        )

    def _order_rejection_reason(self, command: NewOrder) -> str | None:
        if command.participant_id not in self._participants:
            return "unknown_participant"
        if self._states[command.participant_id] is not SessionState.ACTIVE:
            return "session_not_active"
        if command.product_id != self.product.product_id:
            return "unknown_product"
        if self.book.would_self_match(command):
            return "self_match_prevention"
        margin_reason = self._margin_order_rejection_reason(command)
        if margin_reason is not None:
            return margin_reason
        return None

    def _capture_valid_mark(self) -> None:
        """Refresh current and participant-external marks when both sides exist."""

        best_bid = self.book.best_bid
        best_ask = self.book.best_ask
        if best_bid is not None and best_ask is not None:
            self._last_valid_mark = midpoint(best_bid, best_ask)

        snapshot = self.book.snapshot()
        for participant_id, config in self._margin_configs.items():
            if config is None:
                continue
            bid = next(
                (order.price for order in snapshot.bids if order.participant_id != participant_id),
                None,
            )
            ask = next(
                (order.price for order in snapshot.asks if order.participant_id != participant_id),
                None,
            )
            if bid is not None and ask is not None:
                self._last_valid_margin_marks[participant_id] = (
                    midpoint(bid, ask),
                    self.clock.now,
                )

    def _margin_order_rejection_reason(self, command: NewOrder) -> str | None:
        config = self._margin_configs[command.participant_id]
        if config is None:
            return None
        account = self.clearing.snapshot(command.participant_id)
        state = self._margin_states[command.participant_id]
        if state is MarginState.MARGIN_CALL:
            reducing_working = sum(
                order.remaining_quantity
                for order in self.book.orders_for_participant(command.participant_id)
                if order.side is command.side
            )
            reducing = account.position != 0 and (
                (account.position > 0 and command.side is Side.SELL)
                or (account.position < 0 and command.side is Side.BUY)
            )
            if not reducing or reducing_working + command.quantity > abs(account.position):
                return "margin_reduce_only"
            # A margin-called account may always reduce its actual exposure;
            # it must not be trapped merely because it is still below IM.
            return None

        same_side_working = sum(
            order.remaining_quantity
            for order in self.book.orders_for_participant(command.participant_id)
            if order.side is command.side
        )
        projected_position = account.position + command.side.signed(same_side_working + command.quantity)
        mark = self._last_valid_margin_marks.get(command.participant_id)
        if mark is None:
            # Before an external two-sided BBO exists, value the proposed
            # exposure at its submitted limit rather than inventing a mark.
            equity = account.exact_cash + (account.position * command.price * self.product.contract_multiplier)
        else:
            equity = self.clearing.marked_equity_at_mark(command.participant_id, mark=mark[0])
        if equity < abs(projected_position) * config.initial_margin_per_contract:
            return "insufficient_initial_margin"
        return None

    def _margin_rejection_diagnostics(
        self, command: NewOrder, reason: str
    ) -> dict[str, int | float | str | bool | None]:
        config = self._margin_configs.get(command.participant_id)
        if config is None or reason not in {
            "insufficient_initial_margin",
            "margin_reduce_only",
        }:
            return {}
        account = self.clearing.snapshot(command.participant_id)
        same_side_working = sum(
            order.remaining_quantity
            for order in self.book.orders_for_participant(command.participant_id)
            if order.side is command.side
        )
        projected_position = account.position + command.side.signed(same_side_working + command.quantity)
        mark = self._last_valid_margin_marks.get(command.participant_id)
        if mark is None:
            equity = account.exact_cash + (account.position * command.price * self.product.contract_multiplier)
            mark_value: Fraction | None = None
            mark_source = "submitted_limit"
            mark_time: int | None = None
        else:
            mark_value, mark_time = mark
            equity = self.clearing.marked_equity_at_mark(command.participant_id, mark=mark_value)
            mark_source = "external_two_sided_bbo"
        return {
            "margin_state": self._margin_states[command.participant_id].value,
            "mark": None if mark_value is None else _json_number(mark_value),
            "mark_source": mark_source,
            "mark_time": mark_time,
            "equity": _json_number(equity),
            "initial_requirement": (abs(account.position) * config.initial_margin_per_contract),
            "projected_position": projected_position,
            "projected_initial_requirement": (abs(projected_position) * config.initial_margin_per_contract),
            "same_side_working_quantity": same_side_working,
            "reduce_only": self._margin_states[command.participant_id] is MarginState.MARGIN_CALL,
        }

    def _evaluate_margin_states(self) -> None:
        """Re-mark every margined account after every book mutation or fill."""

        self._capture_valid_mark()
        for participant_id, config in self._margin_configs.items():
            if config is None:
                continue
            account = self.clearing.snapshot(participant_id)
            position = abs(account.position)
            risk_mark = self._last_valid_margin_marks.get(participant_id)
            if risk_mark is None and position:
                # There is no valid external mark yet, so retain the prior
                # state rather than allowing a participant's own quote to set it.
                continue
            equity = (
                account.exact_cash
                if risk_mark is None
                else self.clearing.marked_equity_at_mark(participant_id, mark=risk_mark[0])
            )
            initial = position * config.initial_margin_per_contract
            maintenance = position * config.maintenance_margin_per_contract
            if equity < maintenance:
                next_state = MarginState.MARGIN_CALL
            elif equity < initial:
                next_state = MarginState.WARNING
            else:
                next_state = MarginState.NORMAL
            previous = self._margin_states[participant_id]
            if previous is next_state:
                continue

            self._margin_states[participant_id] = next_state
            deadline = self.clock.now + config.grace_period if next_state is MarginState.MARGIN_CALL else None
            self._margin_call_deadlines[participant_id] = deadline
            match_event_id = self._next_match_event_id()
            self._append(
                match_event_id,
                EventKind.RISK,
                {
                    "participant_id": participant_id,
                    "state": next_state.value,
                    "previous_state": previous.value,
                    "mark": (None if risk_mark is None else _json_number(risk_mark[0])),
                    "mark_source": (None if risk_mark is None else "external_two_sided_bbo"),
                    "mark_time": None if risk_mark is None else risk_mark[1],
                    "equity": _json_number(equity),
                    "initial_requirement": initial,
                    "maintenance_requirement": maintenance,
                    "reduce_only": next_state is MarginState.MARGIN_CALL,
                    "liquidation_deadline": deadline,
                },
            )
            if next_state is MarginState.MARGIN_CALL:
                changes = self._cancel_participant_orders(participant_id)
                if changes:
                    self._record_book_changes(match_event_id, changes)
                    self._publish_levels(match_event_id)
                    self._capture_valid_mark()
                    # The forced cancellation can alter the external BBO seen
                    # by every other margined account.
                    self._evaluate_margin_states()

    def _process_due_margin_calls(self) -> None:
        if self._processing_margin_calls:
            return
        self._processing_margin_calls = True
        try:
            for participant_id, deadline in tuple(self._margin_call_deadlines.items()):
                if (
                    self._margin_states[participant_id] is MarginState.MARGIN_CALL
                    and deadline is not None
                    and deadline <= self.clock.now
                ):
                    self._liquidate_margin_once(participant_id)
        finally:
            self._processing_margin_calls = False

    def _liquidate_margin_once(self, participant_id: str) -> None:
        """Submit the smallest ordinary crossing order restoring IM if possible."""

        account = self.clearing.snapshot(participant_id)
        if account.position == 0:
            self._evaluate_margin_states()
            return
        snapshot = self.book.snapshot()
        side = Side.SELL if account.position > 0 else Side.BUY
        opposite = snapshot.bids if side is Side.SELL else snapshot.asks
        available = sum(order.remaining_quantity for order in opposite)
        quantity = min(abs(account.position), available)
        if quantity <= 0:
            return
        quantity = self._minimum_margin_liquidation_quantity(participant_id, side, opposite, quantity)
        cumulative = 0
        limit_price = opposite[-1].price
        for order in opposite:
            cumulative += order.remaining_quantity
            if cumulative >= quantity:
                limit_price = order.price
                break
        match_event_id = self._next_match_event_id()
        command = NewOrder(
            participant_id=participant_id,
            client_order_id=f"__margin_liquidation_{match_event_id}",
            side=side,
            price=limit_price,
            quantity=quantity,
            product_id=self.product.product_id,
        )
        result = self.book.submit(command)
        self._append_order_accepted(
            match_event_id,
            command,
            result.order_id,
            liquidation_reason="margin",
        )
        self._record_match_result(match_event_id, result)

    def _minimum_margin_liquidation_quantity(
        self,
        participant_id: str,
        side: Side,
        opposite: tuple,
        maximum_quantity: int,
    ) -> int:
        config = self._margin_configs[participant_id]
        assert config is not None
        risk_mark = self._last_valid_margin_marks.get(participant_id)
        if risk_mark is None:
            return maximum_quantity
        mark = risk_mark[0]
        account = self.clearing.snapshot(participant_id)
        equity = self.clearing.marked_equity_at_mark(participant_id, mark=mark)
        fees_per_contract = Fraction(
            self.product.transaction_fee_per_contract_subunits,
            self.product.cash_subunits_per_tick,
        )
        position = abs(account.position)
        traded = 0
        adjusted_equity = equity
        for order in opposite:
            for _ in range(min(order.remaining_quantity, maximum_quantity - traded)):
                traded += 1
                if side is Side.SELL:
                    adjusted_equity += (order.price - mark) * self.product.contract_multiplier - fees_per_contract
                else:
                    adjusted_equity += (mark - order.price) * self.product.contract_multiplier - fees_per_contract
                if adjusted_equity >= ((position - traded) * config.initial_margin_per_contract):
                    return traded
        return maximum_quantity

    def _set_market_time(self, market_time: int | None) -> None:
        if market_time is not None:
            self.clock.advance_to(market_time)

    def _append(self, match_event_id: str, kind: EventKind, data: dict) -> Event:
        return self.event_log.append(
            market_time=self.clock.now,
            match_event_id=match_event_id,
            kind=kind,
            product_id=self.product.product_id,
            data=data,
        )

    def _next_match_event_id(self) -> str:
        match_event_id = f"M{self._next_match_event_number}"
        self._next_match_event_number += 1
        return match_event_id

    def _events_from(self, index: int) -> tuple[Event, ...]:
        return self.event_log.from_index(index)

    def _require_participant(self, participant_id: str) -> None:
        if participant_id not in self._participants:
            raise KeyError(f"unknown participant: {participant_id}")


def _json_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


__all__ = [
    "AccountMetrics",
    "CommandResult",
    "Exchange",
    "MarginMetrics",
    "MarginState",
    "SessionState",
    "TerminationResult",
]
