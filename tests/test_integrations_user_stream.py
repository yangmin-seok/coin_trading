from __future__ import annotations

from integrations.binance_ws_user import UserEventRouter, normalize_user_event


def test_user_router_handles_execution_report():
    msg = {"e": "executionReport", "E": 123, "s": "BTCUSDT", "i": 1, "X": "FILLED", "S": "BUY", "l": "0.1", "L": "100"}
    event = normalize_user_event(msg)
    out = UserEventRouter().route(event)
    assert out["kind"] == "order_update"
    assert out["status"] == "FILLED"
