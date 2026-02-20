from __future__ import annotations


class SACPolicyAdapter:
    def __init__(self, model) -> None:
        self.model = model

    def act(self, obs, info) -> float:
        action, _ = self.model.predict(obs, deterministic=True)
        return float(action)
