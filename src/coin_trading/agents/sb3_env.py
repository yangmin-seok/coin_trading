from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from env.execution_model import ExecutionModel
from env.trading_env import TradingEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = object
    spaces = None


class GymTradingEnv(gym.Env if hasattr(gym, "Env") else object):
    """Gymnasium wrapper for TradingEnv compatible with Stable-Baselines3."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        candles: pd.DataFrame,
        features: pd.DataFrame,
        execution_model: ExecutionModel,
        seed: int = 0,
        **env_kwargs: Any,
    ) -> None:
        self._seed = seed
        self.env = TradingEnv(candles, features, execution_model, **env_kwargs)
        obs = self.env.reset()
        self.observation_space = (
            spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)
            if spaces is not None
            else None
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) if spaces is not None else None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs = self.env.reset()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = float(np.asarray(action).reshape(-1)[0])
        obs, reward, done, info = self.env.step(action_value)
        return obs, float(reward), bool(done), False, info
