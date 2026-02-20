from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureComputer(Protocol):
    def reset(self) -> None: ...

    def update(self, candle: dict[str, float]) -> dict[str, float]: ...

    def compute_batch(self, candles_df: pd.DataFrame) -> pd.DataFrame: ...
