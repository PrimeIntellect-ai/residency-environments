from __future__ import annotations

import pytest
from alphaverse.episode import Episode
from alphaverse.player import PlayerSession
from alphaverse.profiles import ParticipantSpec
from alphaverse.strategy import StrategyArtifact
from alphaverse.strategy.policy import StrategySourcePolicyError

SOURCE = """
from alphaverse import Side
from alphaverse.strategy import Strategy

class StrategyImpl(Strategy):
    def on_start(self, ctx, event):
        return [ctx.set_timer("quote", fire_at=10)]

    def on_timer(self, ctx, event):
        return [ctx.submit_limit("child-bid", Side.BUY, 99, 2)]
"""


def test_uploaded_strategy_runs_in_child_and_trades_during_wait() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )

    version_id = session.deploy_source(SOURCE)
    session.wait(until=10)

    assert version_id.startswith("sha256:")
    assert session.open_orders()[0]["client_order_id"] == "child-bid"
    session.stop_strategy()


def test_artifact_id_is_content_addressed() -> None:
    from alphaverse.strategy import StrategyArtifact

    first = StrategyArtifact.build(SOURCE)
    second = StrategyArtifact.build(SOURCE)
    changed = StrategyArtifact.build(SOURCE + "\n# changed")
    assert first.version_id == second.version_id
    assert first.version_id != changed.version_id


@pytest.mark.parametrize(
    "source, message",
    [
        ("import os\n", "module imports are not allowed"),
        ("from pathlib import Path\n", "from 'pathlib' import Path is not allowed"),
        ("open('/etc/passwd')\n", "call to 'open' is not allowed"),
        ("(1).__class__\n", "dunder attribute access is not allowed"),
        (
            "from alphaverse import create_populated_scenario\n",
            "from 'alphaverse' import create_populated_scenario is not allowed",
        ),
    ],
)
def test_source_policy_rejects_private_capabilities(source: str, message: str) -> None:
    with pytest.raises(StrategySourcePolicyError, match=message):
        StrategyArtifact.build(source)


def test_source_policy_allows_named_math_helpers() -> None:
    artifact = StrategyArtifact.build(
        "from math import isfinite, sqrt\n"
        "from alphaverse.strategy import Strategy\n"
        "class StrategyImpl(Strategy):\n"
        "    scale = sqrt(4) if isfinite(4) else 1\n"
    )

    assert artifact.version_id.startswith("sha256:")


def test_worker_audit_hook_denies_runtime_file_access() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )
    source = """
from alphaverse.strategy import Strategy

class StrategyImpl(Strategy):
    leaked = open("/etc/passwd").read()
"""

    # Bypass source validation as evaluator-owned seed installation does. The
    # worker's independent runtime guard still denies the file operation.
    with pytest.raises(RuntimeError, match="strategy capability denied: open"):
        session.deploy_trusted_source(source)


def test_successful_source_deployments_are_audited_across_redeploy_and_stop() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )

    version_id = session.deploy_source(SOURCE)
    assert session.strategy_diagnostics() == {
        "deployment_count": 1,
        "unique_strategy_version_count": 1,
        "strategy_stop_count": 0,
        "strategy_fault_count": 0,
    }

    assert session.deploy_source(SOURCE) == version_id
    records = session.deployment_records()
    assert records[0].outcome == "replaced"
    assert records[0].source == SOURCE
    assert records[1].outcome == "active"

    session.stop_strategy()
    records = session.deployment_records()
    assert records[1].outcome == "stopped"
    assert session.strategy_diagnostics() == {
        "deployment_count": 2,
        "unique_strategy_version_count": 1,
        "strategy_stop_count": 1,
        "strategy_fault_count": 0,
    }


def test_invalid_uploaded_entrypoint_is_rejected_during_deploy() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )
    with pytest.raises(RuntimeError, match="failed to initialize"):
        session.deploy_source("raise RuntimeError('bad source')")


def test_callback_fault_stops_automation_without_stopping_market() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )
    source = """
from alphaverse.strategy import Strategy
class StrategyImpl(Strategy):
    def on_start(self, ctx, event):
        raise RuntimeError("callback exploded")
"""
    session.deploy_source(source)
    session.wait(duration=0)

    status = session.strategy_status()
    assert status["active"] is False
    assert "callback exploded" in str(status["fault"])
    assert session.strategy_diagnostics()["strategy_fault_count"] == 1
    assert session.deployment_records()[0].outcome == "faulted"
    session.wait(duration=1)
    assert session.now == 1


def test_uploaded_strategy_alert_is_durable_in_the_player_inbox() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )
    source = """
from alphaverse.strategy import Strategy
class StrategyImpl(Strategy):
    def on_start(self, ctx, event):
        return [ctx.emit_alert("ready", "strategy is live", {"version": 1})]
"""

    session.deploy_source(source)
    session.wait(duration=0)

    alerts = session.alerts()
    assert len(alerts) == 1
    assert alerts[0].envelope.payload == {
        "event_kind": "strategy_alert",
        "code": "ready",
        "message": "strategy is live",
        "data": {"version": 1},
    }
