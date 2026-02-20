from __future__ import annotations

import pandas as pd

from features.online import OnlineFeatureEngine


def compute_offline(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Offline feature compute via replaying online logic from a fresh state."""
    engine = OnlineFeatureEngine()
    return engine.compute_online_by_replay(candles_df)
