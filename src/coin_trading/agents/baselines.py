from __future__ import annotations

import numpy as np


class BaselinePolicy:
    def reset(self) -> None:
        return None

    def act(self, obs, info) -> float:
        raise NotImplementedError


class BuyAndHold(BaselinePolicy):
    def __init__(self) -> None:
        self.done = False

    def reset(self) -> None:
        self.done = False

    def act(self, obs, info) -> float:
        if not self.done:
            self.done = True
            return 1.0
        return 1.0


class MACrossover(BaselinePolicy):
    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        self.fast = fast
        self.slow = slow
        self.prices: list[float] = []

    def reset(self) -> None:
        self.prices = []

    def act(self, obs, info) -> float:
        price = float(info["close"])
        self.prices.append(price)
        if len(self.prices) < self.slow:
            return 0.0
        ma_fast = np.mean(self.prices[-self.fast :])
        ma_slow = np.mean(self.prices[-self.slow :])
        return 1.0 if ma_fast > ma_slow else 0.0


class VolTarget(BaselinePolicy):
    def __init__(self, target_vol: float = 0.2, window: int = 20) -> None:
        self.target_vol = target_vol
        self.window = window
        self.returns: list[float] = []

    def reset(self) -> None:
        self.returns = []

    def act(self, obs, info) -> float:
        ret = float(info.get("logret_1", 0.0) or 0.0)
        self.returns.append(ret)
        if len(self.returns) < self.window:
            return 0.0
        vol = float(np.std(self.returns[-self.window :]))
        if vol <= 0:
            return 0.0
        return float(np.clip(self.target_vol / vol, 0.0, 1.0))
