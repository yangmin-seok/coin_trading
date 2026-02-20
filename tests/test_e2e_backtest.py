from __future__ import annotations

from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv
from features.offline import compute_offline


def test_e2e_backtest_runs(sample_candles):
    feats = compute_offline(sample_candles)
    env = TradingEnv(sample_candles, feats, ExecutionModel())
    env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        _, _, done, _ = env.step(0.5)
        steps += 1
    assert steps > 1
