from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SupportsAct(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: Any, info: dict[str, Any]) -> float: ...


@dataclass(slots=True)
class PolicyWrapper:
    policy: SupportsAct

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, obs: Any, info: dict[str, Any] | None = None) -> float:
        return float(self.policy.act(obs, info or {}))
