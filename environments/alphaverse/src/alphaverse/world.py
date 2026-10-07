"""Seeded latent demand generation for the simulated participant world.

This module generates causes of future order flow, not prices or fills.  Parent
orders remain hidden world state until participant execution policies turn them
into exchange orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Iterable

from alphaverse.models import MarketTime, Quantity, Side

SECOND = 1_000_000_000


class ExecutionStyle(str, Enum):
    """High-level instruction used later by participant execution policies."""

    SCHEDULED = "scheduled"
    PARTICIPATION = "participation"
    PATIENT = "patient"
    MOMENTUM = "momentum"
    BURST = "burst"


class DemandCause(str, Enum):
    """Hidden source class for a change in participant demand.

    The label is diagnostic world metadata only.  It does not affect matching,
    prices, or fills; strategies decide whether and how to trade from their
    own information and state.
    """

    IDIOSYNCRATIC = "idiosyncratic"
    COHORT = "cohort"


class LatentDemandProfile(str, Enum):
    """Versioned signed imbalance applied to alpha-less parent mandates."""

    STRONG_SHORT = "strong-short"
    MODERATE_SHORT = "moderate-short"
    BALANCED = "balanced"
    MODERATE_LONG = "moderate-long"
    STRONG_LONG = "strong-long"

    @property
    def skew(self) -> float:
        return {
            self.STRONG_SHORT: -0.60,
            self.MODERATE_SHORT: -0.30,
            self.BALANCED: 0.0,
            self.MODERATE_LONG: 0.30,
            self.STRONG_LONG: 0.60,
        }[self]


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Bounds for a deterministic window of alpha-less demand mandates."""

    participant_count: int = 20
    parent_order_count: int = 40
    start_time: MarketTime = 0
    end_time: MarketTime = 3_600 * SECOND
    min_quantity: Quantity = 10
    max_quantity: Quantity = 100
    min_duration: MarketTime = 60 * SECOND
    max_duration: MarketTime = 600 * SECOND
    execution_styles: tuple[ExecutionStyle, ...] = (
        ExecutionStyle.SCHEDULED,
        ExecutionStyle.PARTICIPATION,
        ExecutionStyle.PATIENT,
        ExecutionStyle.MOMENTUM,
        ExecutionStyle.BURST,
    )
    participant_prefix: str = "noise"
    rolling_mandates: bool = False
    min_mandate_gap: MarketTime = 0
    max_mandate_gap: MarketTime = 0
    initial_mandate_spread: MarketTime | None = None
    demand_skew: float = 0.0

    def __post_init__(self) -> None:
        # Copy list-like input into an immutable tuple despite the public type.
        styles = tuple(self.execution_styles)
        object.__setattr__(self, "execution_styles", styles)

        if self.participant_count <= 0:
            raise ValueError("participant_count must be positive")
        if self.parent_order_count < 0:
            raise ValueError("parent_order_count must be non-negative")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be positive")
        if self.max_quantity < self.min_quantity:
            raise ValueError("max_quantity must be at least min_quantity")
        if self.min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration must be at least min_duration")
        if self.max_duration > self.end_time - self.start_time:
            raise ValueError("max_duration must fit within the generation window")
        if not styles:
            raise ValueError("execution_styles must not be empty")
        if any(not isinstance(style, ExecutionStyle) for style in styles):
            raise TypeError("execution_styles must contain only ExecutionStyle values")
        if not self.participant_prefix:
            raise ValueError("participant_prefix must not be empty")
        if not isinstance(self.rolling_mandates, bool):
            raise TypeError("rolling_mandates must be a bool")
        if self.min_mandate_gap < 0:
            raise ValueError("min_mandate_gap must be non-negative")
        if self.max_mandate_gap < self.min_mandate_gap:
            raise ValueError("max_mandate_gap must be at least min_mandate_gap")
        if self.initial_mandate_spread is not None:
            if self.initial_mandate_spread < 0:
                raise ValueError("initial_mandate_spread must be non-negative")
            if self.initial_mandate_spread > self.end_time - self.start_time:
                raise ValueError("initial_mandate_spread must fit within the generation window")
        if (
            isinstance(self.demand_skew, bool)
            or not isinstance(self.demand_skew, (int, float))
            or not -1 <= self.demand_skew <= 1
        ):
            raise ValueError("demand_skew must be numeric and between -1 and 1")


@dataclass(frozen=True, slots=True)
class ParentOrder:
    """A hidden alpha-less demand mandate with a finite execution window."""

    participant_id: str
    side: Side
    total_quantity: Quantity
    start_time: MarketTime
    end_time: MarketTime
    execution_style: ExecutionStyle

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")
        if self.total_quantity <= 0:
            raise ValueError("total_quantity must be positive")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not isinstance(self.execution_style, ExecutionStyle):
            raise TypeError("execution_style must be an ExecutionStyle")

    def cumulative_scheduled_quantity(self, at_time: MarketTime) -> Quantity:
        """Integer linear schedule completed by ``at_time``.

        This is an information statistic, not a claim about realized fills.  A
        later execution policy may trade faster or slower according to style and
        market conditions.
        """

        if at_time <= self.start_time:
            return 0
        if at_time >= self.end_time:
            return self.total_quantity
        elapsed = at_time - self.start_time
        duration = self.end_time - self.start_time
        return self.total_quantity * elapsed // duration

    def scheduled_quantity_between(self, start_time: MarketTime, end_time: MarketTime) -> Quantity:
        """Expected remaining quantity scheduled in ``[start_time, end_time]``."""

        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time < start_time:
            raise ValueError("end_time must be at least start_time")
        return self.cumulative_scheduled_quantity(end_time) - self.cumulative_scheduled_quantity(start_time)


@dataclass(frozen=True, slots=True)
class CohortDemandEvent:
    """A persistent signed target-position shift for one participant cohort.

    ``total_target_delta`` is an unsigned magnitude.  The target ramps linearly
    from zero during ``[start_time, end_time]`` and then remains at its full
    signed value.  This is hidden world state, not a prescribed price path.
    """

    event_id: str
    cohort_id: str
    side: Side
    total_target_delta: Quantity
    start_time: MarketTime
    end_time: MarketTime

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str):
            raise TypeError("event_id must be a string")
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not isinstance(self.cohort_id, str):
            raise TypeError("cohort_id must be a string")
        if not self.cohort_id:
            raise ValueError("cohort_id must not be empty")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")
        if isinstance(self.total_target_delta, bool) or not isinstance(self.total_target_delta, int):
            raise TypeError("total_target_delta must be an int")
        if self.total_target_delta <= 0:
            raise ValueError("total_target_delta must be positive")
        if isinstance(self.start_time, bool) or not isinstance(self.start_time, int):
            raise TypeError("start_time must be an int")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if isinstance(self.end_time, bool) or not isinstance(self.end_time, int):
            raise TypeError("end_time must be an int")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")

    def cumulative_target_change(self, at_time: MarketTime) -> int:
        """Signed cumulative target change at ``at_time``.

        Integer arithmetic deliberately makes this deterministic and keeps the
        event usable by strategies that operate in integral contract units.
        """

        if isinstance(at_time, bool) or not isinstance(at_time, int):
            raise TypeError("at_time must be an int")
        if at_time < 0:
            raise ValueError("at_time must be non-negative")
        if at_time <= self.start_time:
            return 0
        if at_time >= self.end_time:
            return self.side.signed(self.total_target_delta)
        elapsed = at_time - self.start_time
        duration = self.end_time - self.start_time
        quantity = self.total_target_delta * elapsed // duration
        return self.side.signed(quantity) if quantity else 0

    def target_change_between(self, start_time: MarketTime, end_time: MarketTime) -> int:
        """Signed target-position change over ``[start_time, end_time]``."""

        if isinstance(start_time, bool) or not isinstance(start_time, int):
            raise TypeError("start_time must be an int")
        if isinstance(end_time, bool) or not isinstance(end_time, int):
            raise TypeError("end_time must be an int")
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time < start_time:
            raise ValueError("end_time must be at least start_time")
        return self.cumulative_target_change(end_time) - self.cumulative_target_change(start_time)


@dataclass(frozen=True, slots=True)
class CohortDemandConfig:
    """Generation bounds for persistent cohort target-position shifts."""

    cohort_count: int = 4
    event_count: int = 8
    start_time: MarketTime = 0
    end_time: MarketTime = 3_600 * SECOND
    min_target_delta: Quantity = 10
    max_target_delta: Quantity = 100
    min_ramp_duration: MarketTime = 60 * SECOND
    max_ramp_duration: MarketTime = 600 * SECOND

    def __post_init__(self) -> None:
        for name in ("cohort_count", "event_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.cohort_count <= 0:
            raise ValueError("cohort_count must be positive")
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")
        for name in (
            "start_time",
            "end_time",
            "min_target_delta",
            "max_target_delta",
            "min_ramp_duration",
            "max_ramp_duration",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if self.min_target_delta <= 0:
            raise ValueError("min_target_delta must be positive")
        if self.max_target_delta < self.min_target_delta:
            raise ValueError("max_target_delta must be at least min_target_delta")
        if self.min_ramp_duration <= 0:
            raise ValueError("min_ramp_duration must be positive")
        if self.max_ramp_duration < self.min_ramp_duration:
            raise ValueError("max_ramp_duration must be at least min_ramp_duration")
        window = self.end_time - self.start_time
        if self.max_ramp_duration > window:
            raise ValueError("max_ramp_duration must fit within the generation window")
        if self.event_count and self.min_ramp_duration > window // self.event_count:
            raise ValueError("min_ramp_duration must fit within every event time bucket")


class CohortDemandProcess:
    """Seeded, immutable schedule of persistent participant-demand shifts."""

    def __init__(self, seed: int, config: CohortDemandConfig | None = None) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed
        self.config = config if config is not None else CohortDemandConfig()
        if not isinstance(self.config, CohortDemandConfig):
            raise TypeError("config must be a CohortDemandConfig")
        self._events = self._generate_events()

    @property
    def events(self) -> tuple[CohortDemandEvent, ...]:
        """Generated persistent demand events in deterministic time order."""

        return self._events

    def generate_events(self) -> tuple[CohortDemandEvent, ...]:
        """Return the stable generated schedule without advancing RNG state."""

        return self._events

    generate = generate_events

    def target_shift(self, cohort_id: str, at_time: MarketTime) -> int:
        """Aggregate signed target shift for one cohort at ``at_time``."""

        if not isinstance(cohort_id, str):
            raise TypeError("cohort_id must be a string")
        if not cohort_id:
            raise ValueError("cohort_id must not be empty")
        if isinstance(at_time, bool) or not isinstance(at_time, int):
            raise TypeError("at_time must be an int")
        if at_time < 0:
            raise ValueError("at_time must be non-negative")
        return sum(event.cumulative_target_change(at_time) for event in self._events if event.cohort_id == cohort_id)

    def future_signed_target_change(
        self,
        at_time: MarketTime,
        horizon: MarketTime,
        cohort_ids: Iterable[str] | None = None,
    ) -> int:
        """Target-position change scheduled over the next horizon.

        ``cohort_ids`` selects a subset of cohorts for private-signal
        construction.  Omitting it aggregates the whole latent demand world.
        """

        if isinstance(at_time, bool) or not isinstance(at_time, int):
            raise TypeError("at_time must be an int")
        if at_time < 0:
            raise ValueError("at_time must be non-negative")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an int")
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        selected = None
        if cohort_ids is not None:
            selected = frozenset(cohort_ids)
            if not all(isinstance(cohort_id, str) and cohort_id for cohort_id in selected):
                raise ValueError("cohort_ids must contain non-empty strings")
        end_time = at_time + horizon
        return sum(
            event.target_change_between(at_time, end_time)
            for event in self._events
            if selected is None or event.cohort_id in selected
        )

    def _generate_events(self) -> tuple[CohortDemandEvent, ...]:
        config = self.config
        if not config.event_count:
            return ()
        window = config.end_time - config.start_time
        cohort_width = max(3, len(str(config.cohort_count)))
        records: list[CohortDemandEvent] = []
        for index in range(config.event_count):
            # Each scheduled event has its own deterministic random stream.
            # Event generation therefore has no mutable RNG coupling to callers
            # or to a prior event's random draws.
            rng = Random((self.seed << 64) ^ ((index + 1) * 0x9E3779B97F4A7C15))
            bucket_start = config.start_time + index * window // config.event_count
            bucket_end = config.start_time + (index + 1) * window // config.event_count
            maximum_duration = min(config.max_ramp_duration, bucket_end - bucket_start)
            duration = rng.randint(config.min_ramp_duration, maximum_duration)
            start_time = rng.randint(bucket_start, bucket_end - duration)
            cohort_number = rng.randint(1, config.cohort_count)
            records.append(
                CohortDemandEvent(
                    event_id=f"cohort-demand-{index + 1:03d}",
                    cohort_id=f"cohort-{cohort_number:0{cohort_width}d}",
                    side=Side.BUY if rng.getrandbits(1) else Side.SELL,
                    total_target_delta=rng.randint(config.min_target_delta, config.max_target_delta),
                    start_time=start_time,
                    end_time=start_time + duration,
                )
            )
        records.sort(key=lambda event: event.start_time)
        return tuple(records)


@dataclass(frozen=True, slots=True)
class LatentValueProcess:
    """Counter-addressable private-value innovations shared by informed traders.

    Every update index maps to exactly one deterministic innovation. Strategies
    can therefore observe different subsets of the same latent path without
    sharing mutable RNG state or making the result depend on callback order.
    """

    seed: int
    normal_step: int
    shock_probability: float
    shock_size: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if isinstance(self.normal_step, bool) or not isinstance(self.normal_step, int):
            raise TypeError("normal_step must be an int")
        if self.normal_step < 0:
            raise ValueError("normal_step must be non-negative")
        if isinstance(self.shock_probability, bool) or not isinstance(self.shock_probability, (int, float)):
            raise TypeError("shock_probability must be numeric")
        if not 0 <= self.shock_probability <= 1:
            raise ValueError("shock_probability must be between zero and one")
        if isinstance(self.shock_size, bool) or not isinstance(self.shock_size, int):
            raise TypeError("shock_size must be an int")
        if self.shock_size < self.normal_step:
            raise ValueError("shock_size must be at least normal_step")

    def innovation(self, update_index: int) -> int:
        """Return the common innovation for a one-based update index."""

        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise TypeError("update_index must be an int")
        if update_index <= 0:
            raise ValueError("update_index must be positive")

        mixed_seed = (self.seed << 64) ^ (update_index * 0x9E3779B97F4A7C15)
        rng = Random(mixed_seed)
        if rng.random() < self.shock_probability:
            direction = 1 if rng.getrandbits(1) else -1
            return direction * self.shock_size
        if self.normal_step:
            return rng.randint(-self.normal_step, self.normal_step)
        return 0


class WorldGenerator:
    """Generate an immutable latent-demand world from an explicit seed."""

    def __init__(self, seed: int, regime: RegimeConfig | None = None) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self.seed = seed
        self.regime = regime if regime is not None else RegimeConfig()
        self._parent_orders = self._generate_parent_orders()

    @property
    def parent_orders(self) -> tuple[ParentOrder, ...]:
        """The generated mandates as an immutable, stable tuple."""

        return self._parent_orders

    def generate_parent_orders(self) -> tuple[ParentOrder, ...]:
        """Return the already-generated deterministic mandate set.

        Repeated calls do not advance an RNG or produce a subtly different world.
        """

        return self._parent_orders

    # Concise alias useful to episode construction code.
    generate = generate_parent_orders

    def future_signed_remaining_flow(
        self,
        at_time: MarketTime,
        horizon: MarketTime,
        parent_orders: Iterable[ParentOrder] | None = None,
    ) -> int:
        """Hidden signed demand expected to execute over the next horizon.

        The statistic uses an integer linear schedule for each mandate.  BUY is
        positive and SELL is negative.  Supplying records explicitly is useful
        for signal construction and tests; the records are never mutated.
        """

        if at_time < 0:
            raise ValueError("at_time must be non-negative")
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        end_time = at_time + horizon
        orders = self._parent_orders if parent_orders is None else parent_orders
        return sum(
            parent.side.signed(parent.scheduled_quantity_between(at_time, end_time))
            for parent in orders
            if parent.scheduled_quantity_between(at_time, end_time) > 0
        )

    # Vocabulary matching the design notes.
    future_flow = future_signed_remaining_flow

    def _generate_parent_orders(self) -> tuple[ParentOrder, ...]:
        rng = Random(self.seed)
        config = self.regime
        participant_width = max(3, len(str(config.participant_count)))
        records: list[ParentOrder] = []

        if config.rolling_mandates:
            return self._generate_rolling_parent_orders(rng, participant_width)

        for index in range(config.parent_order_count):
            duration = rng.randint(config.min_duration, config.max_duration)
            latest_start = config.end_time - duration
            start_time = rng.randint(config.start_time, latest_start)
            participant_number = rng.randint(1, config.participant_count)
            records.append(
                ParentOrder(
                    participant_id=(f"{config.participant_prefix}-{participant_number:0{participant_width}d}"),
                    side=self._mandate_side(
                        index,
                        balanced_buy=bool(rng.getrandbits(1)),
                    ),
                    total_quantity=rng.randint(config.min_quantity, config.max_quantity),
                    start_time=start_time,
                    end_time=start_time + duration,
                    execution_style=rng.choice(config.execution_styles),
                )
            )

        # Stable time ordering is useful for schedulers. Python's sort is stable,
        # so RNG generation order breaks exact-start ties deterministically.
        records.sort(key=lambda parent: parent.start_time)
        return tuple(records)

    def _generate_rolling_parent_orders(self, rng: Random, participant_width: int) -> tuple[ParentOrder, ...]:
        """Generate repeated, non-overlapping mandates for durable institutions.

        The full schedule is deterministic from the scenario seed.  It is small
        compared with market events even for multi-hour sessions, and gives both
        execution policies and informed signals one immutable source of truth.
        First mandates use a short bounded ramp; each account subsequently
        receives another mandate after a seeded idle gap.
        """

        config = self.regime
        records: list[ParentOrder] = []
        window = max(0, config.end_time - config.start_time - config.max_duration)
        # Do not dilute the active institution pool as the configured market
        # horizon grows. A bounded initial ramp gets every durable account into
        # the market early; thereafter its own duration-plus-gap cycle supplies
        # the rolling flow for the rest of the session.
        default_spread = min(window, config.max_mandate_gap)
        spread = (
            default_spread
            if config.initial_mandate_spread is None
            else min(config.initial_mandate_spread, default_spread)
        )

        for participant_number in range(1, config.participant_count + 1):
            participant_id = f"{config.participant_prefix}-{participant_number:0{participant_width}d}"
            bucket_start = config.start_time + (participant_number - 1) * spread // config.participant_count
            bucket_end = config.start_time + participant_number * spread // config.participant_count
            start_time = rng.randint(bucket_start, bucket_end)

            while start_time + config.min_duration <= config.end_time:
                duration = rng.randint(
                    config.min_duration,
                    min(config.max_duration, config.end_time - start_time),
                )
                records.append(
                    ParentOrder(
                        participant_id=participant_id,
                        side=self._mandate_side(
                            len(records),
                            balanced_buy=bool(rng.getrandbits(1)),
                        ),
                        total_quantity=rng.randint(config.min_quantity, config.max_quantity),
                        start_time=start_time,
                        end_time=start_time + duration,
                        execution_style=rng.choice(config.execution_styles),
                    )
                )
                start_time += duration + rng.randint(config.min_mandate_gap, config.max_mandate_gap)

        records.sort(key=lambda parent: parent.start_time)
        return tuple(records)

    def _mandate_side(self, index: int, *, balanced_buy: bool) -> Side:
        """Select one side without coupling profile changes to other RNG draws.

        The zero-skew path preserves the historical seeded side exactly. For a
        nonzero skew, the same counter-addressed draw is used for both signs, so
        positive and negative profiles are exact side mirrors while all other
        mandate attributes remain unchanged.
        """

        skew = float(self.regime.demand_skew)
        if skew == 0:
            return Side.BUY if balanced_buy else Side.SELL
        draw = Random((self.seed << 64) ^ ((index + 1) * 0xD1B54A32D192ED03) ^ 0x94D049BB133111EB).random()
        favored = draw < (1 + abs(skew)) / 2
        if skew > 0:
            return Side.BUY if favored else Side.SELL
        return Side.SELL if favored else Side.BUY


__all__ = [
    "CohortDemandConfig",
    "CohortDemandEvent",
    "CohortDemandProcess",
    "DemandCause",
    "ExecutionStyle",
    "LatentValueProcess",
    "LatentDemandProfile",
    "ParentOrder",
    "RegimeConfig",
    "WorldGenerator",
]
