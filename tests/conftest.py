from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_candles() -> pd.DataFrame:
    n = 120
    open_time = np.arange(n) * 300_000
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = np.full(n, 10.0)
    return pd.DataFrame(
        {
            "open_time": open_time.astype("int64"),
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
            "close_time": (open_time + 299_999).astype("int64"),
        }
    )
