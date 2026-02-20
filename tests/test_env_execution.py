from __future__ import annotations

import numpy as np

from env.execution_model import ExecutionModel
from env.spaces import OBS_COLUMNS
from env.trading_env import TradingEnv
from src.coin_trading.features.offline import compute_offline


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


def test_execution_model_short_entry_sign_and_costs():
    model = ExecutionModel(fee_rate=0.001, slippage_bps=2.0, max_step_change=1.0, min_delta=0.0)
    result = model.execute_target(
        target_pos=-0.5,
        current_pos=0.0,
        cash=10_000,
        position_qty=0.0,
        equity=10_000,
        next_open=100,
    )

    expected_qty = -50.0
    expected_fill = 99.98
    expected_fee = abs(expected_qty * expected_fill) * 0.001
    expected_slippage = abs(expected_qty) * abs(expected_fill - 100)

    assert result.filled_qty < 0
    assert np.isclose(result.filled_qty, expected_qty)
    assert np.isclose(result.fill_price, expected_fill)
    assert np.isclose(result.fee, expected_fee)
    assert np.isclose(result.slippage_cost, expected_slippage)


def test_execution_model_flip_long_to_short_and_short_to_long():
    model = ExecutionModel(max_step_change=0.6, min_delta=0.01)

    long_to_short = model.execute_target(
        target_pos=-1.0,
        current_pos=0.5,
        cash=10_000,
        position_qty=50.0,
        equity=10_000,
        next_open=100,
    )
    assert np.isclose(long_to_short.effective_target_pos, -0.1)
    assert np.isclose(long_to_short.filled_qty, -60.0)

    short_to_long = model.execute_target(
        target_pos=1.0,
        current_pos=-0.5,
        cash=10_000,
        position_qty=-50.0,
        equity=10_000,
        next_open=100,
    )
    assert np.isclose(short_to_long.effective_target_pos, 0.1)
    assert np.isclose(short_to_long.filled_qty, 60.0)


def test_execution_model_respects_min_delta_for_short_transition():
    model = ExecutionModel(max_step_change=0.25, min_delta=0.05)
    result = model.execute_target(
        target_pos=-0.02,
        current_pos=0.0,
        cash=10_000,
        position_qty=0.0,
        equity=10_000,
        next_open=100,
    )
    assert np.isclose(result.effective_target_pos, 0.0)
    assert np.isclose(result.filled_qty, 0.0)


def test_trading_env_negative_action_moves_to_short(sample_candles):
    feats = compute_offline(sample_candles)
    env = TradingEnv(sample_candles, feats, ExecutionModel(max_step_change=1.0, min_delta=0.0))
    env.reset()

    _obs, _reward, done, info = env.step(-0.6)
    assert not done
    assert np.isclose(info["action_target_pos"], -0.6)
    assert np.isclose(info["action_effective_pos"], -0.6)

    next_open = float(sample_candles.loc[1, "open"])
    position_ratio = (env.state.position_qty * next_open) / env.state.equity
    assert position_ratio < 0


def test_trading_env_position_ratio_is_clipped_for_stability(sample_candles):
    feats = compute_offline(sample_candles)
    env = TradingEnv(sample_candles, feats, ExecutionModel())
    env.reset()
    env.state.position_qty = -1_000.0
    env.state.cash = 0.0
    env.state.equity = 10.0

    obs = env._obs()
    position_ratio_idx = OBS_COLUMNS.index("position_ratio")
    assert obs[position_ratio_idx] == -1.0


def test_obs_column_order_is_stable(sample_candles):
    feats = compute_offline(sample_candles)
    shuffled = feats[list(reversed(feats.columns))]
    env = TradingEnv(sample_candles, shuffled, ExecutionModel())
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs) == len(OBS_COLUMNS)
