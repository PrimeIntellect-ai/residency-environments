"""Versioned opponent ecologies for reproducible evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

OpponentControlScope = Literal["static", "knobs", "full_source"]


@dataclass(frozen=True, slots=True)
class OpponentSlot:
    """One persistent account in an evaluation opponent roster."""

    participant_id: str
    seed_offset: int
    seed_profile: Literal["passive", "competitive"]
    control_scope: OpponentControlScope
    framing: Literal["incumbent", "neutral", "arbitrary"] = "neutral"
    controller_agent_key: str | None = None

    def __post_init__(self) -> None:
        if not self.participant_id or self.participant_id == "player":
            raise ValueError("opponent participant_id must be non-empty and non-player")
        if isinstance(self.seed_offset, bool) or not isinstance(self.seed_offset, int):
            raise TypeError("opponent seed_offset must be an int")
        if self.seed_offset <= 0:
            raise ValueError("opponent seed_offset must be positive")
        if self.control_scope == "static" and self.controller_agent_key is not None:
            raise ValueError("a static opponent cannot have a controller agent")
        if self.control_scope != "static" and not self.controller_agent_key:
            raise ValueError("an adaptive opponent requires a controller agent key")


@dataclass(frozen=True, slots=True)
class OpponentRoster:
    """Immutable opponent accounts and controller capabilities for one eval."""

    roster_id: str
    slots: tuple[OpponentSlot, ...] = ()

    def __post_init__(self) -> None:
        if not self.roster_id:
            raise ValueError("roster_id must not be empty")
        slots = tuple(self.slots)
        object.__setattr__(self, "slots", slots)
        participant_ids = [slot.participant_id for slot in slots]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("opponent participant ids must be unique")
        agent_keys = [slot.controller_agent_key for slot in slots if slot.controller_agent_key is not None]
        if len(agent_keys) != len(set(agent_keys)):
            raise ValueError("opponent controller agent keys must be unique")

    @property
    def controlled_slots(self) -> tuple[OpponentSlot, ...]:
        return tuple(slot for slot in self.slots if slot.controller_agent_key)


_ROSTERS: Mapping[str, OpponentRoster] = MappingProxyType(
    {
        "none-v1": OpponentRoster("none-v1"),
        "competitive-static-v1": OpponentRoster(
            "competitive-static-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="competitive",
                    control_scope="static",
                ),
            ),
        ),
        "passive-static-v1": OpponentRoster(
            "passive-static-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="passive",
                    control_scope="static",
                    framing="incumbent",
                ),
            ),
        ),
        "competitive-static-pair-v1": OpponentRoster(
            "competitive-static-pair-v1",
            (
                OpponentSlot(
                    participant_id="prop-a",
                    seed_offset=1_000_003,
                    seed_profile="competitive",
                    control_scope="static",
                ),
                OpponentSlot(
                    participant_id="prop-b",
                    seed_offset=1_000_033,
                    seed_profile="competitive",
                    control_scope="static",
                ),
            ),
        ),
        "competitive-knobs-v1": OpponentRoster(
            "competitive-knobs-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="competitive",
                    control_scope="knobs",
                    controller_agent_key="prop",
                ),
            ),
        ),
        "passive-knobs-v1": OpponentRoster(
            "passive-knobs-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="passive",
                    control_scope="knobs",
                    framing="incumbent",
                    controller_agent_key="prop",
                ),
            ),
        ),
        "competitive-full-source-v1": OpponentRoster(
            "competitive-full-source-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="competitive",
                    control_scope="full_source",
                    controller_agent_key="prop",
                ),
            ),
        ),
        "passive-full-source-v1": OpponentRoster(
            "passive-full-source-v1",
            (
                OpponentSlot(
                    participant_id="prop",
                    seed_offset=1_000_003,
                    seed_profile="passive",
                    control_scope="full_source",
                    framing="incumbent",
                    controller_agent_key="prop",
                ),
            ),
        ),
    }
)


def opponent_roster(roster_id: str) -> OpponentRoster:
    """Resolve a stable roster id or reject an unknown evaluation contract."""

    try:
        return _ROSTERS[roster_id]
    except KeyError as exc:
        raise ValueError(f"unknown opponent roster: {roster_id!r}") from exc


def legacy_prop_roster_id(
    *,
    adaptive_prop: bool,
    seed_profile: str,
    control_scope: str,
    static: bool = False,
) -> str:
    """Map the original one-prop flags onto an explicit roster id."""

    if not adaptive_prop:
        return "none-v1"
    if static:
        return f"{seed_profile}-static-v1"
    if seed_profile == "passive":
        return f"passive-{'knobs' if control_scope == 'knobs' else 'full-source'}-v1"
    if control_scope == "knobs":
        return "competitive-knobs-v1"
    return "competitive-full-source-v1"


__all__ = [
    "OpponentControlScope",
    "OpponentRoster",
    "OpponentSlot",
    "legacy_prop_roster_id",
    "opponent_roster",
]
