from __future__ import annotations

import pandas as pd

from src.coin_trading.features.common import FeatureState, update_features
from src.coin_trading.features.definitions import FEATURE_COLUMNS


class OnlineFeatureEngine:
    def __init__(self) -> None:
        self.state = FeatureState()

    def reset(self) -> None:
        self.state = FeatureState()

    def update(self, candle: dict[str, float]) -> dict[str, float]:
        feats = update_features(self.state, candle)
        return {col: feats[col] for col in FEATURE_COLUMNS}

    def compute_batch(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        self.reset()
        rows = []
        for row in candles_df.itertuples(index=False):
            rows.append(self.update(row._asdict()))
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    def compute_online_by_replay(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("compute_online_by_replay is deprecated; use compute_batch for both offline training and online replay.")
