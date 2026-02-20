from __future__ import annotations

from integrations.binance_ws_market import extract_kline_payload


def test_extract_kline_payload_combined_stream():
    payload = extract_kline_payload({"stream": "btcusdt@kline_5m", "data": {"e": "kline", "k": {"x": True}}})
    assert payload["e"] == "kline"


def test_extract_kline_payload_raw_stream():
    payload = extract_kline_payload({"e": "kline", "k": {"x": True}})
    assert payload["e"] == "kline"
