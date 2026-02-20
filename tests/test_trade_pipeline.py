from __future__ import annotations

from src.coin_trading.pipelines.trade import run


def test_trade_runtime_builds():
    msg = run()
    assert "trade runtime ready" in msg
