from __future__ import annotations

import numpy as np
import pandas as pd

from features.definitions import FEATURE_COLUMNS
from features.offline import compute_offline


def replay_and_compare(candles_df: pd.DataFrame, feature_engine, n_steps: int = 10_000, atol: float = 1e-9) -> None:
    offline = compute_offline(candles_df)
    online = feature_engine.compute_online_by_replay(candles_df)
    if list(offline.columns) != FEATURE_COLUMNS or list(online.columns) != FEATURE_COLUMNS:
        raise AssertionError("feature schema mismatch")

    tail = min(n_steps, len(offline))
    off_tail = offline.tail(tail).reset_index(drop=True)
    on_tail = online.tail(tail).reset_index(drop=True)

    if not off_tail.isna().equals(on_tail.isna()):
        raise AssertionError("NaN positions mismatch")

    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(off_tail[col].to_numpy(), on_tail[col].to_numpy(), atol=atol, equal_nan=True)
