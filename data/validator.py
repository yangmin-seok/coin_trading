from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ValidationReport:
    rows: int
    missing_count: int
    missing_ratio: float
    duplicate_rows_removed: int
    anomaly_count: int
    gaps: list[dict[str, int]]


class DataValidator:
    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        required = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")

        clean = df.sort_values("open_time").copy()
        before = len(clean)
        clean = clean.drop_duplicates(subset=["open_time"], keep="last")
        deduped = before - len(clean)

        if clean.empty:
            return clean, ValidationReport(0, 0, 0.0, deduped, 0, [])

        diffs = clean["open_time"].diff().dropna()
        gap_points = clean.loc[diffs[diffs > self.interval_ms].index, "open_time"]
        gaps = []
        missing_count = 0
        for idx, end_ts in gap_points.items():
            start_ts = int(clean.loc[idx - 1, "open_time"] + self.interval_ms)
            end_gap = int(end_ts - self.interval_ms)
            count = int((end_gap - start_ts) / self.interval_ms + 1)
            missing_count += count
            gaps.append({"gap_start": start_ts, "gap_end": end_gap, "missing_count": count})

        anomaly_mask = (
            (clean["high"] < clean["low"])
            | (clean["open"] < clean["low"])
            | (clean["open"] > clean["high"])
            | (clean["close"] < clean["low"])
            | (clean["close"] > clean["high"])
            | (clean["volume"] < 0)
            | clean[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
        )
        anomaly_count = int(np.sum(anomaly_mask))
        if anomaly_count > 0:
            raise ValueError(f"anomaly rows detected: {anomaly_count}")

        expected_total = int((clean["open_time"].iloc[-1] - clean["open_time"].iloc[0]) / self.interval_ms + 1)
        missing_ratio = float(missing_count / expected_total) if expected_total else 0.0
        return clean.reset_index(drop=True), ValidationReport(
            rows=len(clean),
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            duplicate_rows_removed=deduped,
            anomaly_count=anomaly_count,
            gaps=gaps,
        )
