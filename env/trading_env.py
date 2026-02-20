from __future__ import annotations

import numpy as np
import pandas as pd

from env.execution_model import ExecutionModel
from env.recorder import StepRecorder
from env.reward import compute_reward
from env.spaces import OBS_COLUMNS
from execution.state import PortfolioState


class TradingEnv:
    def __init__(
        self,
        candles: pd.DataFrame,
        features: pd.DataFrame,
        execution_model: ExecutionModel,
        initial_cash: float = 10_000.0,
        lambda_turnover: float = 0.001,
        lambda_dd: float = 0.1,
        dd_limit: float = 0.2,
    ) -> None:
        self.candles = candles.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.execution_model = execution_model
        self.initial_cash = initial_cash
        self.lambda_turnover = lambda_turnover
        self.lambda_dd = lambda_dd
        self.dd_limit = dd_limit
        self.recorder = StepRecorder()
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        start_price = float(self.candles.loc[0, "close"])
        self.state = PortfolioState(cash=self.initial_cash, position_qty=0.0, equity=self.initial_cash, peak_equity=self.initial_cash)
        self.last_action = 0.0
        self.state.mark_to_market(start_price)
        return self._obs()

    def _obs(self) -> np.ndarray:
        feat_row = self.features.loc[self.t].fillna(0.0).to_dict()
        price = float(self.candles.loc[self.t, "close"])
        position_value = self.state.position_qty * price
        equity = max(self.state.equity, 1e-12)

        obs = {k: feat_row.get(k, 0.0) for k in OBS_COLUMNS}
        obs["cash_ratio"] = self.state.cash / equity
        obs["position_ratio"] = position_value / equity
        obs["unrealized_pnl_ratio"] = (self.state.equity - self.initial_cash) / self.initial_cash
        obs["last_action"] = self.last_action
        return np.array([obs[col] for col in OBS_COLUMNS], dtype=np.float64)

    def step(self, action: float):
        if self.t >= len(self.candles) - 2:
            return self._obs(), 0.0, True, {"reason": "end"}
        price_t = float(self.candles.loc[self.t, "close"])
        self.state.mark_to_market(price_t)
        equity_prev = self.state.equity
        current_pos = (self.state.position_qty * price_t / equity_prev) if equity_prev > 0 else 0.0

        next_open = float(self.candles.loc[self.t + 1, "open"])
        result = self.execution_model.execute_target(action, current_pos, self.state.cash, self.state.position_qty, equity_prev, next_open)
        self.state.cash = result.new_cash
        self.state.position_qty = result.new_position_qty
        self.state.mark_to_market(next_open)

        turnover = abs(result.filled_qty * result.fill_price) / max(equity_prev, 1e-12)
        drawdown = 1.0 - (self.state.equity / max(self.state.peak_equity, 1e-12))
        reward = compute_reward(
            self.state.equity,
            equity_prev,
            turnover,
            drawdown,
            self.lambda_turnover,
            self.lambda_dd,
            self.dd_limit,
        )
        self.last_action = action
        self.t += 1

        info = {
            "equity": self.state.equity,
            "cash": self.state.cash,
            "position_qty": self.state.position_qty,
            "position_value": self.state.position_qty * next_open,
            "fee": result.fee,
            "slippage_cost": result.slippage_cost,
            "drawdown": drawdown,
            "peak_equity": self.state.peak_equity,
            "action_target_pos": action,
            "action_effective_pos": current_pos,
            "fill_price": result.fill_price,
            "filled_qty": result.filled_qty,
        }
        self.recorder.record(info)
        return self._obs(), reward, False, info
