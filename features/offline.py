from __future__ import annotations

import pandas as pd

from features.online import OnlineFeatureEngine


def compute_offline(candles_df: pd.DataFrame) -> pd.DataFrame:
    engine = OnlineFeatureEngine()
    return engine.compute_online_by_replay(candles_df)
