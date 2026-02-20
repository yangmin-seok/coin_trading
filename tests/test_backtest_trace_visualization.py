from __future__ import annotations

from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv
from src.coin_trading.features.offline import compute_offline


def test_backtest_trace_artifacts(sample_candles, tmp_path):
    feats = compute_offline(sample_candles)
    env = TradingEnv(sample_candles, feats, ExecutionModel())
    env.reset()

    actions = [1.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.8, 0.1, 0.1]
    for action in actions:
        _, _, done, _ = env.step(action)
        if done:
            break

    artifacts = env.recorder.write_trace_artifacts(tmp_path)
    assert artifacts["csv"].exists()
    assert artifacts["svg"].exists()

    csv_text = artifacts["csv"].read_text(encoding="utf-8")
    assert "signal" in csv_text
    assert any(label in csv_text for label in ["buy", "sell", "hold"])
