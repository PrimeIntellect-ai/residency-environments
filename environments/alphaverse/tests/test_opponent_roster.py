from __future__ import annotations

import pytest
from alphaverse.artifacts import load_manifest
from alphaverse.episode_runtime import EpisodeRuntime, EpisodeRuntimeConfig
from alphaverse.opponent_roster import (
    OpponentRoster,
    OpponentSlot,
    opponent_roster,
)


def test_roster_rejects_duplicate_participants_and_controller_keys() -> None:
    static = OpponentSlot(
        participant_id="a",
        seed_offset=1,
        seed_profile="competitive",
        control_scope="static",
    )
    with pytest.raises(ValueError, match="participant ids"):
        OpponentRoster("duplicate-participant", (static, static))

    first = OpponentSlot(
        participant_id="a",
        seed_offset=1,
        seed_profile="competitive",
        control_scope="knobs",
        controller_agent_key="controller",
    )
    second = OpponentSlot(
        participant_id="b",
        seed_offset=2,
        seed_profile="competitive",
        control_scope="full_source",
        controller_agent_key="controller",
    )
    with pytest.raises(ValueError, match="agent keys"):
        OpponentRoster("duplicate-agent", (first, second))


def test_runtime_instantiates_every_slot_in_a_versioned_roster(tmp_path) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-roster-test",
            capability_token="control-token",
            capture_token="capture-token",
            scenario_seed=7,
            scenario_version="mvp-v1",
            opponent_roster_id="competitive-static-pair-v1",
            max_market_time_ns=5_000_000_000,
            artifact_root=tmp_path,
        )
    )

    assert runtime.opponent_roster is opponent_roster("competitive-static-pair-v1")
    assert set(runtime.participant_credentials) == {"prop-a", "prop-b"}
    assert runtime.account(participant_id="prop-a")["participant_id"] == "prop-a"
    assert runtime.account(participant_id="prop-b")["participant_id"] == "prop-b"


def test_runtime_artifact_cross_marks_hidden_demand_profile(tmp_path) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-profile-artifact",
            capability_token="control-token",
            capture_token="capture-token",
            scenario_seed=7,
            scenario_version="endogenous-mixed-v1",
            latent_demand_profile="strong-long",
            max_market_time_ns=5_000_000_000,
            artifact_root=tmp_path,
        )
    )

    runtime.terminate()
    metadata = load_manifest(tmp_path / "ep-profile-artifact")["metadata"]

    assert metadata["latent_demand_profile"] == "strong-long"
    assert metadata["latent_parent_order_count"] > 0
    assert metadata["latent_parent_gross_quantity"] > 0
    assert metadata["latent_parent_imbalance"] > 0
