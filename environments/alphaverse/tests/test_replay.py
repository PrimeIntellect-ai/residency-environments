from __future__ import annotations

from alphaverse.episode_runtime import EpisodeRuntime, EpisodeRuntimeConfig
from alphaverse.replay import EpisodeReplay


def test_episode_replay_verifies_and_projects_virtual_time(tmp_path) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-replay",
            capability_token="control",
            capture_token="capture",
            scenario_seed=9,
            scenario_version="mvp-v1",
            max_market_time_ns=2_000_000_000,
            time_mode="manual",
            artifact_root=tmp_path,
        )
    )
    runtime.wait(
        duration_ns=2_000_000_000,
        until_ns=None,
        wake_on_alert=False,
    )
    trusted = runtime.sync_terminal()
    assert trusted is not None

    replay = EpisodeReplay(tmp_path / "ep-replay")

    assert replay.receipt()["terminal_event_sequence"] > 0
    assert replay.frame_at(1_000_000_000)["market_time_ns"] == 0
    assert replay.frame_at(2_000_000_000)["market_time_ns"] == 2_000_000_000
    events = list(replay.events())
    assert events[-1]["sequence"] == replay.manifest["terminal_event_sequence"]
