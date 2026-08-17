from __future__ import annotations

import pytest
from alphaverse.models import Side
from alphaverse.strategy import (
    ActionBatch,
    EmitAlert,
    EmitLog,
    InputEnvelope,
    InputKind,
    LogLevel,
    SetTimer,
    SubmitLimitOrder,
)


def _event(kind: InputKind = InputKind.MARKET) -> InputEnvelope:
    return InputEnvelope(
        session_id="S1",
        strategy_instance_id="I1",
        event_id="E1",
        kind=kind,
        exchange_time=10,
        available_at=12,
        source_event_seq=7,
        payload={"nested": [1, {"ok": True}]},
    )


def test_input_envelope_round_trip_is_stable_and_immutable() -> None:
    event = _event()
    restored = InputEnvelope.from_json(event.to_json())
    assert restored.to_json() == event.to_json()
    assert restored.payload["nested"] == (1, {"ok": True})
    with pytest.raises(TypeError):
        restored.payload["new"] = 1  # type: ignore[index]


def test_prevalidated_payload_can_be_shared_safely_between_envelopes() -> None:
    source = {"nested": [1, {"ok": True}]}
    payload = InputEnvelope.freeze_payload(source)
    first = InputEnvelope("S", "I1", "E", InputKind.MARKET, 10, 10, 1, payload)
    second = InputEnvelope("S", "I2", "E", InputKind.MARKET, 10, 10, 1, payload)

    source["nested"].append(2)
    assert first.payload is payload
    assert second.payload is payload
    assert first.payload["nested"] == (1, {"ok": True})
    with pytest.raises(TypeError):
        first.payload["new"] = 1  # type: ignore[index]


def test_available_at_cannot_precede_exchange_time() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        InputEnvelope("S", "I", "E", InputKind.MARKET, 10, 9, 1)


def test_action_batch_round_trip_preserves_order_and_signed_prices() -> None:
    batch = ActionBatch(
        [
            SubmitLimitOrder("c1", Side.BUY, -5, 2),
            SetTimer("refresh", 100),
            EmitLog(LogLevel.INFO, "quoted", {"price": -5}),
            EmitAlert("quote_ready", "A quote is resting", {"price": -5}),
        ]
    )
    restored = ActionBatch.from_json(batch.to_json())
    assert restored == batch
    assert restored.to_json() == batch.to_json()


def test_submit_limit_requires_positive_quantity() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        SubmitLimitOrder("c1", Side.SELL, 10, 0)


def test_decode_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unknown action type"):
        ActionBatch.from_json('{"actions":[{"type":"hack"}]}')


def test_emit_alert_requires_concise_json_compatible_content() -> None:
    alert = EmitAlert("edge", "signal crossed threshold", {"score": 1.25})
    assert alert.data == {"score": 1.25}
    with pytest.raises(ValueError, match="64"):
        EmitAlert("x" * 65, "message")
    with pytest.raises(ValueError, match="4096"):
        EmitAlert("edge", "message", {"payload": "x" * 4_097})
