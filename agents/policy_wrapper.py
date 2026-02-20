from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PolicyLike(Protocol):
    def act(self, obs, info) -> float: ...


@dataclass(slots=True)
class BoundedPolicy:
    policy: PolicyLike
    low: float = 0.0
    high: float = 1.0

    def act(self, obs, info) -> float:
        value = float(self.policy.act(obs, info))
        return float(max(self.low, min(self.high, value)))
