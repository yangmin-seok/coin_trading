from __future__ import annotations

from execution.marketdata import CandleClosedEvent
from pipelines.trade import build_runtime, process_market_event


def test_process_market_event_produces_decision_payload():
    runtime = build_runtime()
    ev = CandleClosedEvent(
        symbol="BTCUSDT",
        interval="5m",
        open_time_ms=0,
        close_time_ms=299999,
        o=100.0,
        h=101.0,
        l=99.0,
        c=100.5,
        v=10.0,
        num_trades=1,
        is_final=True,
        source="ws",
    )
    out = process_market_event(runtime, ev)
    assert "target" in out
    assert "approved" in out
