from __future__ import annotations

import pandas as pd

from src.coin_trading.features.offline import compute_offline


def compute_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Single feature interface shared by offline training and online replay parity."""
    return compute_offline(candles_df)
