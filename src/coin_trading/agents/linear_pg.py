from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LinearPolicyGradient:
    """Tiny numpy-only policy gradient model for offline train pipeline smoke training."""

    obs_dim: int
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    weights: np.ndarray = field(init=False)
    bias: float = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.weights = self.rng.normal(0.0, 0.05, size=self.obs_dim)
        self.bias = 0.0

    def _forward(self, obs: np.ndarray) -> float:
        z = float(np.dot(self.weights, obs) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def act(self, obs: np.ndarray) -> float:
        return float(np.clip(self._forward(obs), 0.0, 1.0))

    def train_step(self, obs: np.ndarray, reward: float, lr: float = 1e-2) -> None:
        action = self._forward(obs)
        grad_scale = reward * action * (1.0 - action)
        self.weights += lr * grad_scale * obs
        self.bias += lr * grad_scale
