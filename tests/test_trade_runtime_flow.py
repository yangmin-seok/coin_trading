from __future__ import annotations

from execution.marketdata import CandleClosedEvent
from pipelines.trade import build_runtime, process_candle_event


def test_process_candle_event_fills_buy_order(tmp_path):
    runtime = build_runtime()
    runtime.metrics.path = tmp_path / "runtime_metrics.jsonl"
    event = CandleClosedEvent(
        symbol="BTCUSDT",
        interval="5m",
        open_time_ms=0,
        close_time_ms=299_999,
        o=100.0,
        h=101.0,
        l=99.0,
        c=100.0,
        v=10.0,
        num_trades=1,
        is_final=True,
        source="ws",
    )
    out = process_candle_event(runtime, event)
    assert out["status"] == "filled"
    assert runtime.state.position_qty > 0
