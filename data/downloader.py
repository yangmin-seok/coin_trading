from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


CANDLE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(slots=True)
class HistoricalDownloader:
    exchange: str = "binance"
    market: str = "spot"

    def fetch(self, symbol: str, interval: str, start_utc: datetime, end_utc: datetime, source_mode: str = "auto") -> pd.DataFrame:
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start_utc/end_utc must be timezone-aware UTC datetime")
        if start_utc.tzinfo != timezone.utc or end_utc.tzinfo != timezone.utc:
            raise ValueError("timestamps must be UTC")
        if end_utc <= start_utc:
            raise ValueError("end_utc must be greater than start_utc")
        return pd.DataFrame(columns=CANDLE_COLUMNS)


def normalize_time_unit(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    out = df.copy()
    for col in ["open_time", "close_time"]:
        if not out.empty and out[col].max() > 10**14:
            out[col] = out[col] // 1000
    if not out.empty:
        remainder = out["open_time"] % interval_ms
        if not (remainder == remainder.iloc[0]).all():
            raise ValueError("inconsistent open_time alignment for interval")
    return out
