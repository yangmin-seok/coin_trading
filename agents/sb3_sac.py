from __future__ import annotations


class SACPolicy:
    """Optional SB3 SAC policy shim.

    This project does not require stable-baselines3 at runtime.
    If the package is unavailable, calling `act` raises a clear error.
    """

    def __init__(self) -> None:
        self._model = None

    def reset(self) -> None:
        return None

    def load(self, model_path: str) -> None:
        try:
            from stable_baselines3 import SAC
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("stable-baselines3 is required to load SAC models") from exc
        self._model = SAC.load(model_path)

    def act(self, obs, info) -> float:
        if self._model is None:
            raise RuntimeError("SAC model is not loaded. Call load(model_path) first.")
        action, _ = self._model.predict(obs, deterministic=True)
        return float(action)
