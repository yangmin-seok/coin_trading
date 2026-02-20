from __future__ import annotations

from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv
from features.offline import compute_offline


def test_hold_like_behavior_is_stable(sample_candles):
    feats = compute_offline(sample_candles)
    env = TradingEnv(sample_candles, feats, ExecutionModel(fee_rate=0.0, slippage_bps=0.0, max_step_change=0.0, min_delta=1.0))
    env.reset()
    equities = []
    done = False
    while not done:
        _, _, done, info = env.step(0.0)
        if not done:
            equities.append(info["equity"])
    assert max(equities) - min(equities) < 1e-6
