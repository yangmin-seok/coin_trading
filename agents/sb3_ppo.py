from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SB3LikeModel(Protocol):
    def predict(self, observation: Any, deterministic: bool = True): ...


@dataclass(slots=True)
class PPOPolicyAdapter:
    model: SB3LikeModel
    deterministic: bool = True

    def reset(self) -> None:
        return None

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> float:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return float(action)
