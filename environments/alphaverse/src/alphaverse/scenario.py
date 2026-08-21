"""Deterministic populated market used by local trials and evaluation tasks.

The scenario deliberately composes the same :class:`~alphaverse.episode.Episode`
and strategy interfaces used by player code.  There is no privileged order-book
mutation: opening liquidity, maker refreshes, and latent noise mandates all reach
the exchange as ordinary strategy actions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from alphaverse.episode import Episode
from alphaverse.exchange import Exchange
from alphaverse.informed import FutureFlowInformedTrader
from alphaverse.models import Side
from alphaverse.profiles import (
    InformationGrant,
    ParticipantSpec,
    RiskLimits,
    TechnologyProfile,
)
from alphaverse.reference_strategies import (
    AdaptiveMarketMaker,
    InventoryAwareMarketMaker,
    LatentValueTrader,
    PersistentNoiseTrader,
    ReservationDemandTrader,
    RollingNoiseExecutor,
)
from alphaverse.strategy import InputEnvelope, Strategy, StrategyContext
from alphaverse.world import (
    SECOND,
    CohortDemandConfig,
    CohortDemandProcess,
    ExecutionStyle,
    LatentDemandProfile,
    LatentValueProcess,
    ParentOrder,
    RegimeConfig,
    WorldGenerator,
)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Configuration for the small, continuously active MVP market."""

    seed: int = 0
    session_id: str = "alphaverse-mvp"
    reference_price: int = 10_000
    opening_half_spread: int = 3
    opening_quantity: int = 500
    opening_liquidity_duration: int | None = None
    starting_cash: int = 10_000_000
    level_depth: int = 10

    maker_count: int = 4
    maker_refresh_interval: int = SECOND // 4
    maker_quote_quantity: int = 80
    maker_base_half_spread: int = 2
    maker_inventory_skew_base: float = 0.005
    maker_latency_step: int = SECOND // 1_000

    adaptive_maker_count: int = 0
    adaptive_quote_quantity_fraction: float = 0.25
    adaptive_inventory_skew_floor: float = 0.0
    adaptive_fill_pressure_per_unit: float = 0.02
    adaptive_fill_pressure_half_life: int = 2 * SECOND
    adaptive_maximum_fill_skew: float = 4.0
    adaptive_markout_horizon: int = 3 * SECOND
    adaptive_markout_learning_rate: float = 0.25
    adaptive_toxicity_widening_multiplier: float = 1.0
    adaptive_toxicity_half_life: int = 15 * SECOND
    adaptive_maximum_toxicity_widening: int = 6
    adaptive_minimum_quote_fraction: float = 0.25
    adaptive_directional_toxicity: bool = False

    noise_order_count: int = 16
    warmup_duration: int = 2 * SECOND
    active_duration: int = 3 * 60 * SECOND
    noise_min_quantity: int = 20
    noise_max_quantity: int = 100
    noise_min_duration: int = 15 * SECOND
    noise_max_duration: int = 60 * SECOND
    noise_slice_interval: int = 5 * SECOND
    noise_max_slice_quantity: int = 10
    noise_min_mandate_gap: int = 60 * SECOND
    noise_max_mandate_gap: int = 120 * SECOND
    latent_demand_profile: LatentDemandProfile = LatentDemandProfile.BALANCED

    persistent_noise_count: int = 6
    persistent_noise_min_interval: int = 2 * SECOND
    persistent_noise_max_interval: int = 5 * SECOND
    persistent_noise_min_quantity: int = 1
    persistent_noise_max_quantity: int = 7
    persistent_noise_aggressive_probability: float = 0.65
    persistent_noise_side_persistence: float = 0.65
    persistent_noise_soft_inventory_limit: int = 40

    informed_trader_count: int = 10
    informed_update_interval: int = 15 * SECOND
    informed_signal_horizon: int = 30 * SECOND
    informed_signal_loadings: tuple[float, ...] = (
        1.0,
        0.8,
        1.0,
        0.4,
        0.9,
        0.6,
        1.1,
        0.75,
        1.0,
        0.3,
    )
    informed_signal_observation_probabilities: tuple[float, ...] = (
        0.75,
        0.4,
        0.6,
        0.3,
        0.5,
        0.35,
        0.7,
        0.45,
        0.55,
        0.25,
    )
    informed_signal_noise_floor: int = 2
    informed_signal_noise_multipliers: tuple[float, ...] = (
        0.25,
        1.5,
        0.75,
        3.0,
        1.0,
        2.5,
        0.5,
        1.5,
        0.5,
        3.0,
    )
    informed_minimum_signal: int = 5
    informed_minimum_parent_quantity: int = 2
    informed_maximum_parent_quantity: int = 10
    informed_quantity_per_signal_unit: float = 0.12
    informed_metaorder_duration: int = 12 * SECOND
    informed_slice_interval: int = 3 * SECOND
    informed_execution_styles: tuple[ExecutionStyle, ...] = (
        ExecutionStyle.SCHEDULED,
        ExecutionStyle.PARTICIPATION,
        ExecutionStyle.PATIENT,
        ExecutionStyle.MOMENTUM,
        ExecutionStyle.BURST,
    )

    latent_value_trader_count: int = 0
    latent_value_update_interval: int = 15 * SECOND
    latent_value_order_quantity: int = 25
    latent_value_normal_step: int = 2
    latent_value_shock_probability: float = 0.05
    latent_value_shock_size: int = 8
    latent_value_signal_observation_probability: float = 0.5
    latent_value_signal_noise_step: int = 1
    latent_value_signal_proportional_noise_multiplier: float = 2.0
    latent_value_signal_proportional_noise_fraction: float = 1.0

    # Optional endogenous-demand ecology. These participants have heterogeneous
    # slow market anchors and elastic target inventories. Cohort events change
    # participant demand, never an exchange-level or authoritative price.
    reservation_trader_count: int = 0
    reservation_cohort_count: int = 4
    reservation_update_interval: int = SECOND
    reservation_anchor_half_life: int = 2 * 60 * SECOND
    reservation_anchor_half_life_multipliers: tuple[float, ...] = (1.0,)
    reservation_anchor_dispersion_ticks: int = 8
    reservation_price_elasticity: float = 2.0
    reservation_price_elasticity_multipliers: tuple[float, ...] = (1.0,)
    reservation_base_target_spread: int = 20
    reservation_clip_quantity: int = 5
    reservation_position_tolerance: int = 2
    cohort_demand_event_count: int = 0
    cohort_demand_start_delay: int = 30 * SECOND
    cohort_demand_end_buffer: int = 120 * SECOND
    cohort_demand_min_target_delta: int = 40
    cohort_demand_max_target_delta: int = 80
    cohort_demand_min_ramp_duration: int = 20 * SECOND
    cohort_demand_max_ramp_duration: int = 45 * SECOND

    def __post_init__(self) -> None:
        profile = self.latent_demand_profile
        if isinstance(profile, str):
            try:
                profile = LatentDemandProfile(profile)
            except ValueError as exc:
                raise ValueError(f"unknown latent_demand_profile: {profile!r}") from exc
            object.__setattr__(self, "latent_demand_profile", profile)
        elif not isinstance(profile, LatentDemandProfile):
            raise TypeError("latent_demand_profile must be a LatentDemandProfile")
        signal_noise_multipliers = tuple(self.informed_signal_noise_multipliers)
        observation_probabilities = tuple(self.informed_signal_observation_probabilities)
        signal_loadings = tuple(self.informed_signal_loadings)
        execution_styles = tuple(self.informed_execution_styles)
        reservation_half_life_multipliers = tuple(self.reservation_anchor_half_life_multipliers)
        reservation_elasticity_multipliers = tuple(self.reservation_price_elasticity_multipliers)
        object.__setattr__(
            self,
            "informed_signal_loadings",
            signal_loadings,
        )
        object.__setattr__(
            self,
            "informed_signal_observation_probabilities",
            observation_probabilities,
        )
        object.__setattr__(
            self,
            "informed_signal_noise_multipliers",
            signal_noise_multipliers,
        )
        object.__setattr__(
            self,
            "informed_execution_styles",
            execution_styles,
        )
        object.__setattr__(
            self,
            "reservation_anchor_half_life_multipliers",
            reservation_half_life_multipliers,
        )
        object.__setattr__(
            self,
            "reservation_price_elasticity_multipliers",
            reservation_elasticity_multipliers,
        )
        _require_non_negative("seed", self.seed)
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.opening_liquidity_duration is not None:
            _require_positive(
                "opening_liquidity_duration",
                self.opening_liquidity_duration,
            )
        if isinstance(self.reference_price, bool) or not isinstance(self.reference_price, int):
            raise TypeError("reference_price must be an int")
        for name in (
            "opening_half_spread",
            "opening_quantity",
            "starting_cash",
            "maker_refresh_interval",
            "maker_quote_quantity",
            "maker_base_half_spread",
            "adaptive_fill_pressure_half_life",
            "adaptive_markout_horizon",
            "adaptive_toxicity_half_life",
            "noise_order_count",
            "warmup_duration",
            "active_duration",
            "noise_min_quantity",
            "noise_max_quantity",
            "noise_min_duration",
            "noise_max_duration",
            "noise_slice_interval",
            "noise_max_slice_quantity",
            "noise_min_mandate_gap",
            "noise_max_mandate_gap",
            "persistent_noise_min_interval",
            "persistent_noise_max_interval",
            "persistent_noise_min_quantity",
            "persistent_noise_max_quantity",
            "persistent_noise_soft_inventory_limit",
            "informed_update_interval",
            "informed_signal_horizon",
            "informed_minimum_signal",
            "informed_minimum_parent_quantity",
            "informed_maximum_parent_quantity",
            "informed_metaorder_duration",
            "informed_slice_interval",
            "latent_value_update_interval",
            "latent_value_order_quantity",
            "latent_value_shock_size",
            "reservation_cohort_count",
            "reservation_update_interval",
            "reservation_anchor_half_life",
            "reservation_clip_quantity",
            "cohort_demand_min_target_delta",
            "cohort_demand_max_target_delta",
            "cohort_demand_min_ramp_duration",
            "cohort_demand_max_ramp_duration",
        ):
            _require_positive(name, getattr(self, name))
        _require_non_negative(
            "persistent_noise_count",
            self.persistent_noise_count,
        )
        _require_non_negative(
            "latent_value_trader_count",
            self.latent_value_trader_count,
        )
        _require_non_negative(
            "informed_trader_count",
            self.informed_trader_count,
        )
        _require_non_negative(
            "reservation_trader_count",
            self.reservation_trader_count,
        )
        _require_non_negative(
            "reservation_anchor_dispersion_ticks",
            self.reservation_anchor_dispersion_ticks,
        )
        _require_non_negative(
            "reservation_base_target_spread",
            self.reservation_base_target_spread,
        )
        _require_non_negative(
            "reservation_position_tolerance",
            self.reservation_position_tolerance,
        )
        _require_non_negative(
            "cohort_demand_event_count",
            self.cohort_demand_event_count,
        )
        if self.cohort_demand_event_count and not self.reservation_trader_count:
            raise ValueError("cohort demand events require reservation traders")
        _require_non_negative(
            "cohort_demand_start_delay",
            self.cohort_demand_start_delay,
        )
        _require_non_negative(
            "cohort_demand_end_buffer",
            self.cohort_demand_end_buffer,
        )
        _require_non_negative(
            "informed_signal_noise_floor",
            self.informed_signal_noise_floor,
        )
        _require_non_negative("latent_value_normal_step", self.latent_value_normal_step)
        _require_non_negative(
            "latent_value_signal_noise_step",
            self.latent_value_signal_noise_step,
        )
        _require_non_negative("level_depth", self.level_depth)
        _require_non_negative("maker_count", self.maker_count)
        _require_non_negative("maker_latency_step", self.maker_latency_step)
        _require_non_negative("adaptive_maker_count", self.adaptive_maker_count)
        _require_non_negative(
            "adaptive_maximum_toxicity_widening",
            self.adaptive_maximum_toxicity_widening,
        )
        if not isinstance(self.adaptive_directional_toxicity, bool):
            raise TypeError("adaptive_directional_toxicity must be a bool")
        if not isinstance(self.maker_inventory_skew_base, (int, float)) or isinstance(
            self.maker_inventory_skew_base, bool
        ):
            raise TypeError("maker_inventory_skew_base must be numeric")
        if self.maker_inventory_skew_base < 0:
            raise ValueError("maker_inventory_skew_base must be non-negative")
        for name in (
            "adaptive_fill_pressure_per_unit",
            "adaptive_inventory_skew_floor",
            "adaptive_maximum_fill_skew",
            "adaptive_toxicity_widening_multiplier",
            "informed_quantity_per_signal_unit",
            "latent_value_signal_proportional_noise_multiplier",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not isinstance(self.reservation_price_elasticity, (int, float))
            or isinstance(self.reservation_price_elasticity, bool)
            or not 0 < self.reservation_price_elasticity < float("inf")
        ):
            raise ValueError("reservation_price_elasticity must be finite and positive")
        for name, values in (
            (
                "reservation_anchor_half_life_multipliers",
                reservation_half_life_multipliers,
            ),
            (
                "reservation_price_elasticity_multipliers",
                reservation_elasticity_multipliers,
            ),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < float("inf")
                for value in values
            ):
                raise ValueError(f"{name} must contain finite positive values")
        if not signal_loadings:
            raise ValueError("informed_signal_loadings must not be empty")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < float("inf")
            for value in signal_loadings
        ):
            raise ValueError("informed_signal_loadings must be finite and non-negative")
        if self.informed_maximum_parent_quantity < self.informed_minimum_parent_quantity:
            raise ValueError("informed_maximum_parent_quantity must be at least its minimum")
        if not signal_noise_multipliers:
            raise ValueError("informed_signal_noise_multipliers must not be empty")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < float("inf")
            for value in signal_noise_multipliers
        ):
            raise ValueError("informed_signal_noise_multipliers must be finite and non-negative")
        if not observation_probabilities:
            raise ValueError("informed_signal_observation_probabilities must not be empty")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1
            for value in observation_probabilities
        ):
            raise ValueError("informed_signal_observation_probabilities must be between zero and one")
        if not execution_styles:
            raise ValueError("informed_execution_styles must not be empty")
        if any(not isinstance(style, ExecutionStyle) for style in execution_styles):
            raise TypeError("informed_execution_styles must contain ExecutionStyle values")
        for name in (
            "adaptive_markout_learning_rate",
            "adaptive_minimum_quote_fraction",
            "adaptive_quote_quantity_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.noise_max_quantity < self.noise_min_quantity:
            raise ValueError("noise_max_quantity must be at least noise_min_quantity")
        if self.noise_max_duration < self.noise_min_duration:
            raise ValueError("noise_max_duration must be at least noise_min_duration")
        if self.noise_max_duration > self.active_duration:
            raise ValueError("noise_max_duration must fit within active_duration")
        if self.noise_max_mandate_gap < self.noise_min_mandate_gap:
            raise ValueError("noise_max_mandate_gap must be at least noise_min_mandate_gap")
        if self.persistent_noise_max_interval < self.persistent_noise_min_interval:
            raise ValueError("persistent_noise_max_interval must be at least its minimum")
        if self.persistent_noise_max_quantity < self.persistent_noise_min_quantity:
            raise ValueError("persistent_noise_max_quantity must be at least its minimum")
        if self.cohort_demand_max_target_delta < self.cohort_demand_min_target_delta:
            raise ValueError("cohort_demand_max_target_delta must be at least its minimum")
        if self.cohort_demand_max_ramp_duration < self.cohort_demand_min_ramp_duration:
            raise ValueError("cohort_demand_max_ramp_duration must be at least its minimum")
        if self.cohort_demand_event_count and (
            self.cohort_demand_start_delay + self.cohort_demand_max_ramp_duration + self.cohort_demand_end_buffer
            > self.active_duration
        ):
            raise ValueError("cohort demand start delay, ramp, and end buffer must fit within active_duration")
        for name in (
            "persistent_noise_aggressive_probability",
            "persistent_noise_side_persistence",
            "latent_value_shock_probability",
            "latent_value_signal_observation_probability",
            "latent_value_signal_proportional_noise_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


MVP_SCENARIO_VERSION = "mvp-v1"
ENDOGENOUS_MIXED_SCENARIO_VERSION = "endogenous-mixed-v1"

# The research cell and evaluation task share these exact values so a
# calibration change cannot silently leave the task environment behind.
ENDOGENOUS_MIXED_OVERRIDES: Mapping[str, object] = MappingProxyType(
    {
        "opening_quantity": 1_250,
        "opening_liquidity_duration": 5 * SECOND,
        "maker_count": 10,
        "informed_minimum_parent_quantity": 1,
        "informed_maximum_parent_quantity": 5,
        "informed_quantity_per_signal_unit": 0.06,
        "reservation_trader_count": 16,
        "reservation_cohort_count": 4,
        "reservation_update_interval": 5 * SECOND,
        "reservation_anchor_half_life": 5 * 60 * SECOND,
        "reservation_anchor_half_life_multipliers": (0.1, 0.4, 1.0, 4.0),
        "reservation_anchor_dispersion_ticks": 4,
        "reservation_price_elasticity": 4.0,
        "reservation_price_elasticity_multipliers": (0.5, 1.0, 1.5, 1.0),
        "reservation_base_target_spread": 0,
        "reservation_position_tolerance": 2,
        "cohort_demand_event_count": 1,
        "cohort_demand_min_target_delta": 160,
        "cohort_demand_max_target_delta": 320,
    }
)


def scenario_config_for_version(
    scenario_version: str,
    **episode_overrides: object,
) -> ScenarioConfig:
    """Build a scenario from a stable named profile plus episode overrides."""

    if scenario_version == MVP_SCENARIO_VERSION:
        profile: dict[str, object] = {}
    elif scenario_version == ENDOGENOUS_MIXED_SCENARIO_VERSION:
        profile = dict(ENDOGENOUS_MIXED_OVERRIDES)
    else:
        raise ValueError(f"unknown scenario_version: {scenario_version!r}")
    profile.update(episode_overrides)
    return ScenarioConfig(**profile)


@dataclass(slots=True)
class PopulatedScenario:
    """A built episode plus its immutable latent world and participant metadata."""

    config: ScenarioConfig
    episode: Episode
    world: WorldGenerator
    cohort_demand_process: CohortDemandProcess
    latent_value_process: LatentValueProcess
    parent_orders: tuple[ParentOrder, ...]
    maker_ids: tuple[str, ...]
    adaptive_maker_ids: tuple[str, ...]
    noise_ids: tuple[str, ...]
    persistent_noise_ids: tuple[str, ...]
    informed_ids: tuple[str, ...]
    latent_value_ids: tuple[str, ...]
    reservation_demand_ids: tuple[str, ...]
    reservation_aggregate_price_elasticity: float

    @property
    def now(self) -> int:
        return self.episode.now

    def run_until(self, market_time: int) -> int:
        """Advance the deterministic world to an absolute market timestamp."""

        return self.episode.run_until(market_time)

    def wait(self, duration: int) -> int:
        """Let the market run without requiring the caller to emit any tokens."""

        _require_non_negative("duration", duration)
        return self.run_until(self.now + duration)

    @property
    def latent_parent_signed_quantity(self) -> int:
        return sum(parent.side.signed(parent.total_quantity) for parent in self.parent_orders)

    @property
    def latent_parent_gross_quantity(self) -> int:
        return sum(parent.total_quantity for parent in self.parent_orders)

    @property
    def latent_parent_imbalance(self) -> float:
        gross = self.latent_parent_gross_quantity
        return self.latent_parent_signed_quantity / gross if gross else 0.0


class _OpeningLiquidity(Strategy):
    """One-time deep quotes that break the empty-book initialization cycle."""

    def __init__(
        self,
        reference_price: int,
        half_spread: int,
        quantity: int,
        *,
        duration: int | None = None,
        timer_id: str = "opening-liquidity-withdrawal",
    ) -> None:
        self.reference_price = reference_price
        self.half_spread = half_spread
        self.quantity = quantity
        self.duration = duration
        self.timer_id = timer_id
        self._pending_clients = {"opening-bid", "opening-ask"}
        self._order_ids: set[str] = set()

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        actions = [
            ctx.submit_limit(
                "opening-bid",
                Side.BUY,
                self.reference_price - self.half_spread,
                self.quantity,
            ),
            ctx.submit_limit(
                "opening-ask",
                Side.SELL,
                self.reference_price + self.half_spread,
                self.quantity,
            ),
        ]
        if self.duration is not None:
            actions.append(ctx.set_timer(self.timer_id, fire_at=ctx.now + self.duration))
        return actions

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        if payload.get("event_kind") == "order_accepted":
            client_order_id = payload.get("client_order_id")
            order_id = payload.get("order_id")
            if (
                isinstance(client_order_id, str)
                and client_order_id in self._pending_clients
                and isinstance(order_id, str)
            ):
                self._pending_clients.discard(client_order_id)
                self._order_ids.add(order_id)
        elif payload.get("event_kind") == "cancel_accepted":
            order_id = payload.get("order_id")
            if isinstance(order_id, str):
                self._order_ids.discard(order_id)
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("timer_id") != self.timer_id:
            return None
        return [ctx.cancel(order_id) for order_id in sorted(self._order_ids)]


def create_populated_scenario(
    config: ScenarioConfig | None = None,
    *,
    run_warmup: bool = True,
) -> PopulatedScenario:
    """Build a seeded market and optionally advance it through warmup.

    The opening participant supplies robust initial depth.  Heterogeneous makers
    then take over regular quoting, while persistent noise institutions execute
    repeated hidden mandates after the warmup boundary.
    """

    cfg = config or ScenarioConfig()
    episode = Episode(
        Exchange(level_depth=cfg.level_depth),
        session_id=cfg.session_id,
    )

    episode.add_strategy(
        _participant_spec(
            "opening-liquidity",
            cfg,
            seed=cfg.seed,
            max_abs_position=10 * cfg.opening_quantity,
            max_order_quantity=cfg.opening_quantity,
        ),
        _OpeningLiquidity(
            cfg.reference_price,
            cfg.opening_half_spread,
            cfg.opening_quantity,
            duration=cfg.opening_liquidity_duration,
        ),
    )

    maker_ids: list[str] = []
    for index in range(cfg.maker_count):
        participant_id = f"maker-{index + 1:02d}"
        maker_ids.append(participant_id)
        # Repeat four calibrated variants when population studies request more
        # makers.  Letting size and risk grow with the absolute index would
        # confound participant count with increasingly gigantic accounts.
        variant_index = index % 4
        # Different refresh rates, spreads, sizes, risk tolerance, and transport
        # latency make the maker population heterogeneous without another RNG.
        refresh_jitter = max(1, cfg.maker_refresh_interval // 5)
        refresh_interval = cfg.maker_refresh_interval + (index % 3) * refresh_jitter
        quote_quantity = cfg.maker_quote_quantity + (variant_index * max(1, cfg.maker_quote_quantity // 4))
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 1_000 + index,
                market_data_latency=(index % 3) * cfg.maker_latency_step,
                order_entry_latency=(index % 2) * cfg.maker_latency_step,
                max_abs_position=quote_quantity * (5 + variant_index),
                max_order_quantity=quote_quantity,
            ),
            InventoryAwareMarketMaker(
                refresh_interval=refresh_interval,
                quote_quantity=quote_quantity,
                base_half_spread=cfg.maker_base_half_spread + index % 3,
                volatility_multiplier=0.5 + 0.25 * (index % 3),
                inventory_skew_per_unit=(cfg.maker_inventory_skew_base * (variant_index + 1)),
            ),
        )

    adaptive_maker_ids: list[str] = []
    for index in range(cfg.adaptive_maker_count):
        participant_id = f"adaptive-maker-{index + 1:02d}"
        adaptive_maker_ids.append(participant_id)
        variant_index = index % 4
        refresh_jitter = max(1, cfg.maker_refresh_interval // 5)
        refresh_interval = cfg.maker_refresh_interval + (index % 3) * refresh_jitter
        paired_quote_quantity = cfg.maker_quote_quantity + (variant_index * max(1, cfg.maker_quote_quantity // 4))
        quote_quantity = max(
            1,
            math.ceil(paired_quote_quantity * cfg.adaptive_quote_quantity_fraction),
        )
        inventory_limit = quote_quantity * (5 + variant_index)
        variant_multiplier = 0.75 + 0.25 * (index % 3)
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 5_000 + index,
                market_data_latency=(index % 3) * cfg.maker_latency_step,
                order_entry_latency=(index % 2) * cfg.maker_latency_step,
                max_abs_position=inventory_limit,
                max_order_quantity=quote_quantity,
            ),
            AdaptiveMarketMaker(
                refresh_interval=refresh_interval,
                initial_refresh_delay=max(1, refresh_interval // 2),
                quote_quantity=quote_quantity,
                base_half_spread=cfg.maker_base_half_spread + index % 3,
                volatility_multiplier=0.5 + 0.25 * (index % 3),
                inventory_skew_per_unit=(
                    max(
                        cfg.adaptive_inventory_skew_floor,
                        cfg.maker_inventory_skew_base * (variant_index + 1),
                    )
                ),
                inventory_soft_limit=max(1, inventory_limit // 2),
                minimum_quote_fraction=cfg.adaptive_minimum_quote_fraction,
                fill_pressure_per_unit=(cfg.adaptive_fill_pressure_per_unit * variant_multiplier),
                fill_pressure_half_life=cfg.adaptive_fill_pressure_half_life,
                maximum_fill_skew=cfg.adaptive_maximum_fill_skew,
                markout_horizon=cfg.adaptive_markout_horizon,
                markout_learning_rate=cfg.adaptive_markout_learning_rate,
                toxicity_widening_multiplier=(cfg.adaptive_toxicity_widening_multiplier * variant_multiplier),
                toxicity_half_life=cfg.adaptive_toxicity_half_life,
                maximum_toxicity_widening=(cfg.adaptive_maximum_toxicity_widening),
                directional_toxicity=cfg.adaptive_directional_toxicity,
            ),
        )

    regime = RegimeConfig(
        participant_count=cfg.noise_order_count,
        parent_order_count=0,
        start_time=cfg.warmup_duration,
        end_time=cfg.warmup_duration + cfg.active_duration,
        min_quantity=cfg.noise_min_quantity,
        max_quantity=cfg.noise_max_quantity,
        min_duration=cfg.noise_min_duration,
        max_duration=cfg.noise_max_duration,
        participant_prefix="noise",
        rolling_mandates=True,
        min_mandate_gap=cfg.noise_min_mandate_gap,
        max_mandate_gap=cfg.noise_max_mandate_gap,
        demand_skew=cfg.latent_demand_profile.skew,
    )
    world = WorldGenerator(cfg.seed, regime)
    latent_value_process = LatentValueProcess(
        seed=cfg.seed + 40_000,
        normal_step=cfg.latent_value_normal_step,
        shock_probability=cfg.latent_value_shock_probability,
        shock_size=cfg.latent_value_shock_size,
    )
    if cfg.cohort_demand_event_count:
        cohort_demand_config = CohortDemandConfig(
            cohort_count=cfg.reservation_cohort_count,
            event_count=cfg.cohort_demand_event_count,
            start_time=(cfg.warmup_duration + cfg.cohort_demand_start_delay),
            end_time=(cfg.warmup_duration + cfg.active_duration - cfg.cohort_demand_end_buffer),
            min_target_delta=cfg.cohort_demand_min_target_delta,
            max_target_delta=cfg.cohort_demand_max_target_delta,
            min_ramp_duration=cfg.cohort_demand_min_ramp_duration,
            max_ramp_duration=cfg.cohort_demand_max_ramp_duration,
        )
    else:
        # Keep the default ecology byte-for-byte compatible even in unit tests
        # whose synthetic market-time scale is much smaller than one second.
        cohort_demand_config = CohortDemandConfig(
            cohort_count=cfg.reservation_cohort_count,
            event_count=0,
            start_time=0,
            end_time=1,
            min_target_delta=1,
            max_target_delta=1,
            min_ramp_duration=1,
            max_ramp_duration=1,
        )
    cohort_demand_process = CohortDemandProcess(
        seed=cfg.seed + 50_000,
        config=cohort_demand_config,
    )

    parent_orders = world.parent_orders
    mandates_by_participant: defaultdict[str, list[ParentOrder]] = defaultdict(list)
    for parent in parent_orders:
        mandates_by_participant[parent.participant_id].append(parent)
    noise_ids = tuple(f"noise-{index:03d}" for index in range(1, cfg.noise_order_count + 1))
    for index, participant_id in enumerate(noise_ids):
        mandates = tuple(mandates_by_participant[participant_id])
        if not mandates:
            raise RuntimeError("rolling mandate generation omitted an institution")
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 10_000 + index,
                market_data_latency=(index % 4) * cfg.maker_latency_step,
                order_entry_latency=(index % 3) * cfg.maker_latency_step,
                max_abs_position=50 * cfg.noise_max_quantity,
                max_order_quantity=cfg.noise_max_slice_quantity,
            ),
            RollingNoiseExecutor(
                mandates,
                slice_interval=cfg.noise_slice_interval,
                max_slice_quantity=cfg.noise_max_slice_quantity,
            ),
        )

    reservation_demand_ids: list[str] = []
    reservation_elasticities = tuple(
        cfg.reservation_price_elasticity
        * cfg.reservation_price_elasticity_multipliers[
            (index // cfg.reservation_cohort_count) % len(cfg.reservation_price_elasticity_multipliers)
        ]
        for index in range(cfg.reservation_trader_count)
    )
    cohort_member_counts = {
        cohort_index: sum(
            participant_index % cfg.reservation_cohort_count == cohort_index
            for participant_index in range(cfg.reservation_trader_count)
        )
        for cohort_index in range(cfg.reservation_cohort_count)
    }
    for index in range(cfg.reservation_trader_count):
        participant_id = f"reservation-demand-{index + 1:02d}"
        reservation_demand_ids.append(participant_id)
        cohort_index = index % cfg.reservation_cohort_count
        cohort_width = max(3, len(str(cfg.reservation_cohort_count)))
        cohort_id = f"cohort-{cohort_index + 1:0{cohort_width}d}"
        denominator = max(1, cfg.reservation_trader_count - 1)
        centered_index = 2 * index - (cfg.reservation_trader_count - 1)
        anchor_offset = round(cfg.reservation_anchor_dispersion_ticks * centered_index / denominator)
        base_target = round(cfg.reservation_base_target_spread * centered_index / denominator)
        event_capacity = cfg.cohort_demand_max_target_delta
        position_capacity = abs(base_target) + event_capacity + 10 * cfg.reservation_clip_quantity
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 45_000 + index,
                market_data_latency=(index % 4) * cfg.maker_latency_step,
                order_entry_latency=(index % 3) * cfg.maker_latency_step,
                max_abs_position=position_capacity,
                max_order_quantity=cfg.reservation_clip_quantity,
            ),
            ReservationDemandTrader(
                demand_process=cohort_demand_process,
                cohort_id=cohort_id,
                initial_anchor=cfg.reference_price + anchor_offset,
                update_interval=cfg.reservation_update_interval,
                anchor_half_life=max(
                    1,
                    round(
                        cfg.reservation_anchor_half_life
                        * cfg.reservation_anchor_half_life_multipliers[
                            (index // cfg.reservation_cohort_count) % len(cfg.reservation_anchor_half_life_multipliers)
                        ]
                    ),
                ),
                price_elasticity=reservation_elasticities[index],
                base_target_position=base_target,
                target_loading=(1.0 / cohort_member_counts[cohort_index]),
                clip_quantity=cfg.reservation_clip_quantity,
                position_tolerance=cfg.reservation_position_tolerance,
            ),
        )

    persistent_noise_ids: list[str] = []
    for index in range(cfg.persistent_noise_count):
        participant_id = f"recurring-noise-{index + 1:03d}"
        persistent_noise_ids.append(participant_id)
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 20_000 + index,
                market_data_latency=(index % 4) * cfg.maker_latency_step,
                order_entry_latency=(index % 3) * cfg.maker_latency_step,
                max_abs_position=2 * cfg.persistent_noise_soft_inventory_limit,
                max_order_quantity=cfg.persistent_noise_max_quantity,
            ),
            PersistentNoiseTrader(
                min_interval=cfg.persistent_noise_min_interval,
                max_interval=cfg.persistent_noise_max_interval,
                min_quantity=cfg.persistent_noise_min_quantity,
                max_quantity=cfg.persistent_noise_max_quantity,
                aggressive_probability=(cfg.persistent_noise_aggressive_probability),
                side_persistence=cfg.persistent_noise_side_persistence,
                soft_inventory_limit=(cfg.persistent_noise_soft_inventory_limit),
            ),
        )

    informed_ids: list[str] = []
    for index in range(cfg.informed_trader_count):
        participant_id = f"informed-{index + 1:02d}"
        informed_ids.append(participant_id)
        maximum_parent_quantity = cfg.informed_maximum_parent_quantity
        quality_index = (index * 3 + cfg.seed) % len(cfg.informed_signal_noise_multipliers)
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 25_000 + index,
                market_data_latency=(index % 4) * cfg.maker_latency_step,
                order_entry_latency=(index % 3) * cfg.maker_latency_step,
                max_abs_position=50 * maximum_parent_quantity,
                max_order_quantity=maximum_parent_quantity,
                extra_information_grants=frozenset((InformationGrant.FUTURE_FLOW_SIGNAL,)),
            ),
            FutureFlowInformedTrader(
                world=world,
                parent_orders=None,
                update_interval=cfg.informed_update_interval,
                signal_horizon=cfg.informed_signal_horizon,
                signal_loading=cfg.informed_signal_loadings[quality_index % len(cfg.informed_signal_loadings)],
                signal_observation_probability=(
                    cfg.informed_signal_observation_probabilities[
                        quality_index % len(cfg.informed_signal_observation_probabilities)
                    ]
                ),
                signal_noise_multiplier=(
                    cfg.informed_signal_noise_multipliers[quality_index % len(cfg.informed_signal_noise_multipliers)]
                ),
                signal_noise_floor=cfg.informed_signal_noise_floor,
                minimum_signal=cfg.informed_minimum_signal,
                minimum_parent_quantity=(cfg.informed_minimum_parent_quantity),
                maximum_parent_quantity=maximum_parent_quantity,
                quantity_per_signal_unit=(cfg.informed_quantity_per_signal_unit),
                metaorder_duration=cfg.informed_metaorder_duration,
                slice_interval=cfg.informed_slice_interval,
                execution_style=cfg.informed_execution_styles[index % len(cfg.informed_execution_styles)],
            ),
        )

    latent_value_ids: list[str] = []
    proportional_noise_trader_count = math.ceil(
        cfg.latent_value_trader_count * cfg.latent_value_signal_proportional_noise_fraction
    )
    for index in range(cfg.latent_value_trader_count):
        participant_id = f"latent-value-{index + 1:02d}"
        latent_value_ids.append(participant_id)
        episode.add_strategy(
            _participant_spec(
                participant_id,
                cfg,
                seed=cfg.seed + 30_000 + index,
                market_data_latency=index * cfg.maker_latency_step,
                order_entry_latency=index * cfg.maker_latency_step,
                max_abs_position=50 * cfg.latent_value_order_quantity,
                max_order_quantity=cfg.latent_value_order_quantity,
                extra_information_grants=frozenset((InformationGrant.LATENT_VALUE_SIGNAL,)),
            ),
            LatentValueTrader(
                initial_value=cfg.reference_price,
                update_interval=cfg.latent_value_update_interval,
                order_quantity=cfg.latent_value_order_quantity,
                normal_step=cfg.latent_value_normal_step,
                shock_probability=cfg.latent_value_shock_probability,
                shock_size=cfg.latent_value_shock_size,
                signal_process=latent_value_process,
                signal_observation_probability=(cfg.latent_value_signal_observation_probability),
                signal_noise_step=cfg.latent_value_signal_noise_step,
                signal_proportional_noise_multiplier=(
                    cfg.latent_value_signal_proportional_noise_multiplier
                    if index >= cfg.latent_value_trader_count - proportional_noise_trader_count
                    else 0.0
                ),
            ),
        )

    scenario = PopulatedScenario(
        config=cfg,
        episode=episode,
        world=world,
        cohort_demand_process=cohort_demand_process,
        latent_value_process=latent_value_process,
        parent_orders=parent_orders,
        maker_ids=tuple(maker_ids),
        adaptive_maker_ids=tuple(adaptive_maker_ids),
        noise_ids=noise_ids,
        persistent_noise_ids=tuple(persistent_noise_ids),
        informed_ids=tuple(informed_ids),
        latent_value_ids=tuple(latent_value_ids),
        reservation_demand_ids=tuple(reservation_demand_ids),
        reservation_aggregate_price_elasticity=sum(reservation_elasticities),
    )
    if run_warmup:
        scenario.run_until(cfg.warmup_duration)
    return scenario


def _participant_spec(
    participant_id: str,
    config: ScenarioConfig,
    *,
    seed: int,
    max_abs_position: int,
    max_order_quantity: int,
    market_data_latency: int = 0,
    order_entry_latency: int = 0,
    extra_information_grants: frozenset[InformationGrant] = frozenset(),
) -> ParticipantSpec:
    return ParticipantSpec(
        participant_id=participant_id,
        strategy_version_id=f"scenario-v1:{participant_id}",
        account_starting_cash=config.starting_cash,
        information_grants=frozenset(
            (
                InformationGrant.PUBLIC_MARKET,
                InformationGrant.OWN_EXECUTIONS,
                *extra_information_grants,
            )
        ),
        technology=TechnologyProfile(
            market_data_latency=market_data_latency,
            order_entry_latency=order_entry_latency,
            # The built-in strategies consume the level feed, not MBO. Keeping
            # the unused entitlement off avoids constructing no-op callbacks.
            mbo_entitled=False,
        ),
        risk=RiskLimits(
            max_abs_position=max_abs_position,
            max_order_quantity=max_order_quantity,
            max_live_orders=1_000,
            max_actions_per_callback=100,
        ),
        seed=seed,
    )


def _require_non_negative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: object) -> None:
    _require_non_negative(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "ENDOGENOUS_MIXED_OVERRIDES",
    "ENDOGENOUS_MIXED_SCENARIO_VERSION",
    "MVP_SCENARIO_VERSION",
    "LatentDemandProfile",
    "PopulatedScenario",
    "ScenarioConfig",
    "create_populated_scenario",
    "scenario_config_for_version",
]
