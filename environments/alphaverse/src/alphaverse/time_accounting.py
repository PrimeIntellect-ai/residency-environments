"""Trusted bridges from agent activity to deterministic episode time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from alphaverse.player import PlayerSession, WaitResult


class TimeMode(str, Enum):
    MANUAL = "manual"
    WALL = "wall"
    TOKENS = "tokens"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cached_input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class TokenTimeConfig:
    ns_per_turn: int = 0
    ns_per_input_token: int = 0
    ns_per_cached_input_token: int = 0
    ns_per_output_token: int = 0

    def __post_init__(self) -> None:
        for name in (
            "ns_per_turn",
            "ns_per_input_token",
            "ns_per_cached_input_token",
            "ns_per_output_token",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def market_delta(self, usage: TokenUsage) -> int:
        return (
            self.ns_per_turn
            + usage.input_tokens * self.ns_per_input_token
            + usage.cached_input_tokens * self.ns_per_cached_input_token
            + usage.output_tokens * self.ns_per_output_token
        )


@dataclass(frozen=True, slots=True)
class TimeAdvance:
    reason: str
    from_time: int
    to_time: int
    idempotency_key: str | None = None

    @property
    def duration(self) -> int:
        return self.to_time - self.from_time


class EpisodeTimeController:
    """Advance one session from wall time, trusted token usage, or explicit waits."""

    def __init__(
        self,
        session: PlayerSession,
        *,
        mode: TimeMode = TimeMode.MANUAL,
        wall_time_scale: float = 1.0,
        wall_quantum_ns: int = 1_000_000,
        token_config: TokenTimeConfig | None = None,
        max_market_time: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(mode, TimeMode):
            raise TypeError("mode must be a TimeMode")
        if wall_time_scale < 0:
            raise ValueError("wall_time_scale must be non-negative")
        if wall_quantum_ns <= 0:
            raise ValueError("wall_quantum_ns must be positive")
        if max_market_time is not None and max_market_time < session.now:
            raise ValueError("max_market_time cannot precede current market time")
        self.session = session
        self.mode = mode
        self.wall_time_scale = wall_time_scale
        self.wall_quantum_ns = wall_quantum_ns
        self.token_config = token_config or TokenTimeConfig()
        self.max_market_time = max_market_time
        self._monotonic_ns = monotonic_ns
        # Wall charging begins at the first player operation, not episode
        # allocation. Container/tool startup is evaluator overhead and must not
        # consume the player's market-time budget.
        self._wall_anchor: int | None = None
        self._market_anchor = session.now
        self._committed_turns: set[str] = set()
        self._advances: list[TimeAdvance] = []
        self._paused = False
        self._advance_limit: int | None = None

    @property
    def advances(self) -> tuple[TimeAdvance, ...]:
        return tuple(self._advances)

    @property
    def committed_turn_count(self) -> int:
        return len(self._committed_turns)

    @property
    def paused(self) -> bool:
        """Whether agent activity is currently exempt from market-time charging."""

        return self._paused

    @property
    def advance_limit(self) -> int | None:
        """Current absolute boundary that no time source may cross."""

        return self._advance_limit

    def set_advance_limit(self, market_time: int | None) -> None:
        """Clamp every time source to an upcoming market-session boundary."""

        if market_time is not None:
            if isinstance(market_time, bool) or not isinstance(market_time, int):
                raise TypeError("market_time must be an int or None")
            if market_time < self.session.now:
                raise ValueError("advance limit cannot precede current market time")
        self._advance_limit = market_time

    def pause(self) -> None:
        """Freeze market charging while agents use a research intermission."""

        self._paused = True
        self._wall_anchor = None
        self._market_anchor = self.session.now

    def resume(self) -> None:
        """Resume charging from a fresh anchor after an intermission."""

        self._paused = False
        self._wall_anchor = None
        self._market_anchor = self.session.now

    def before_operation(self) -> int:
        """Charge elapsed agent wall time immediately before a service operation."""

        if self._paused or self.mode is not TimeMode.WALL:
            return self.session.now
        now = self._monotonic_ns()
        if self._wall_anchor is None:
            self._wall_anchor = now
            self._market_anchor = self.session.now
            return self.session.now
        elapsed = max(0, now - self._wall_anchor)
        scaled = int(elapsed * self.wall_time_scale)
        quantized = scaled - scaled % self.wall_quantum_ns
        target = max(self.session.now, self._market_anchor + quantized)
        return self._advance_to(target, reason="wall_time")

    def commit_turn(self, turn_id: str, usage: TokenUsage) -> int:
        """Charge one trusted interception result exactly once."""

        if self.mode is not TimeMode.TOKENS:
            raise RuntimeError("turn commits require tokens time mode")
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        if turn_id in self._committed_turns:
            return 0
        self._committed_turns.add(turn_id)
        if self._paused:
            return 0
        delta = self.token_config.market_delta(usage)
        before = self.session.now
        after = self._advance_to(
            before + delta,
            reason="model_turn",
            idempotency_key=turn_id,
        )
        return after - before

    def voluntary_wait(
        self,
        duration: int,
        *,
        interrupt_on_alert: bool = False,
    ) -> int | WaitResult:
        if isinstance(duration, bool) or not isinstance(duration, int):
            raise TypeError("duration must be an int")
        if duration < 0:
            raise ValueError("duration must be non-negative")
        before = self.session.now
        if self._paused:
            return (
                WaitResult(
                    market_time=before,
                    target_market_time=before + duration,
                    interrupted_by_alert=False,
                )
                if interrupt_on_alert
                else before
            )
        target = before + duration
        if self.max_market_time is not None:
            target = min(target, self.max_market_time)
        if self._advance_limit is not None:
            target = min(target, self._advance_limit)
        result = self.session.wait(
            until=target,
            interrupt_on_alert=interrupt_on_alert,
        )
        after = result.market_time if isinstance(result, WaitResult) else result
        if after != before:
            self._advances.append(TimeAdvance("voluntary_wait", before, after))
        self._wall_anchor = self._monotonic_ns()
        self._market_anchor = self.session.now
        return result

    def _advance_to(
        self,
        target: int,
        *,
        reason: str,
        idempotency_key: str | None = None,
    ) -> int:
        if self.max_market_time is not None:
            target = min(target, self.max_market_time)
        if self._advance_limit is not None:
            target = min(target, self._advance_limit)
        target = max(target, self.session.now)
        before = self.session.now
        # A zero-duration wait is still meaningful: it drains actions and feed
        # deliveries already scheduled at the current virtual timestamp.
        self.session.wait(until=target)
        if target != before:
            self._advances.append(TimeAdvance(reason, before, target, idempotency_key))
        return self.session.now


__all__ = [
    "EpisodeTimeController",
    "TimeAdvance",
    "TimeMode",
    "TokenTimeConfig",
    "TokenUsage",
]
