from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agents.baselines import BaselinePolicy, BuyAndHold, MACrossover, VolTarget
from agents.sb3_ppo import PPOPolicy
from agents.sb3_sac import SACPolicy


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs, info) -> float: ...


@dataclass(slots=True)
class PolicyWrapper:
    policy: Policy

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs, info) -> float:
        action = float(self.policy.act(obs, info))
        return max(0.0, min(1.0, action))


def create_policy(name: str) -> PolicyWrapper:
    normalized = name.strip().lower()
    if normalized in {"buy_and_hold", "buyandhold", "bh"}:
        policy: Policy = BuyAndHold()
    elif normalized in {"ma", "ma_crossover", "macrossover"}:
        policy = MACrossover()
    elif normalized in {"vol", "vol_target", "voltarget"}:
        policy = VolTarget()
    elif normalized in {"ppo", "sb3_ppo"}:
        policy = PPOPolicy()
    elif normalized in {"sac", "sb3_sac"}:
        policy = SACPolicy()
    else:
        raise ValueError(f"Unsupported policy: {name}")
    return PolicyWrapper(policy)


__all__ = ["Policy", "PolicyWrapper", "create_policy", "BaselinePolicy"]
