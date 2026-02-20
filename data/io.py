from __future__ import annotations

from pathlib import Path

import pandas as pd


PARTITION_COLUMNS = ["exchange", "market", "symbol", "interval", "date"]


def build_partition_path(root: str | Path, exchange: str, market: str, symbol: str, interval: str, date: str) -> Path:
    return Path(root) / f"exchange={exchange}" / f"market={market}" / f"symbol={symbol}" / f"interval={interval}" / f"date={date}"


def write_candles_parquet(df: pd.DataFrame, root: str | Path, exchange: str, market: str, symbol: str, interval: str) -> list[Path]:
    written: list[Path] = []
    df = df.sort_values("open_time")
    df = df.assign(date=pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d"))
    for date, chunk in df.groupby("date"):
        path = build_partition_path(root, exchange, market, symbol, interval, date)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "part-000.parquet"
        chunk.drop(columns=["date"]).to_parquet(file_path, index=False)
        written.append(file_path)
    return written


def read_candles_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values("open_time").reset_index(drop=True)
