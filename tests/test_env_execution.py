from __future__ import annotations

import numpy as np

from env.execution_model import ExecutionModel
from env.spaces import OBS_COLUMNS
from env.trading_env import TradingEnv
from features.offline import compute_offline


def test_execution_model_limits_delta():
    model = ExecutionModel(max_step_change=0.1, min_delta=0.01)
    result = model.execute_target(
        target_pos=1.0,
        current_pos=0.0,
        cash=10_000,
        position_qty=0.0,
        equity=10_000,
        next_open=100,
    )
    assert result.filled_qty > 0
    assert result.new_position_qty == result.filled_qty


def test_obs_column_order_is_stable(sample_candles):
    feats = compute_offline(sample_candles)
    shuffled = feats[list(reversed(feats.columns))]
    env = TradingEnv(sample_candles, shuffled, ExecutionModel())
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == len(OBS_COLUMNS)
