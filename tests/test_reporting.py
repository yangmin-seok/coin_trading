from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.coin_trading.pipelines.reporting import (
    create_benchmark_comparison,
    create_common_risk_plots,
    create_split_equity_curves,
    detect_overfit,
    write_trade_stats_report,
)


def _sample_trace(n: int = 20) -> pd.DataFrame:
    rows = []
    base_ts = 1_700_000_000_000
    equity = 1000.0
    for i in range(n):
        equity *= 1.001 if i % 3 else 0.999
        rows.append(
            {
                "open_time": base_ts + i * 60_000,
                "equity": equity,
                "drawdown": max(0.0, (1010.0 - equity) / 1010.0),
                "action_effective_pos": 1.0 if i % 2 else 0.0,
                "filled_qty": 0.1 if i % 5 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_reporting_artifacts_created(tmp_path: Path):
    traces = {"train": _sample_trace(), "valid": _sample_trace(), "test": _sample_trace()}

    split_files = create_split_equity_curves(tmp_path, traces)
    assert len(split_files) == 3
    for rel in split_files:
        assert (tmp_path / rel).exists()

    risk = create_common_risk_plots(tmp_path, traces)
    assert (tmp_path / risk["drawdown_curve_png"]).exists()
    assert (tmp_path / risk["monthly_returns_heatmap_png"]).exists()

    report = write_trade_stats_report(tmp_path, traces["test"], overfit_warning=True)
    report_path = tmp_path / report
    assert report_path.exists()
    assert "Overfitting Warning" in report_path.read_text(encoding="utf-8")


def test_benchmark_and_overfit(tmp_path: Path):
    candles = pd.DataFrame({"close": [100, 101, 100.5, 103, 102], "open_time": [1, 2, 3, 4, 5]})
    rel = create_benchmark_comparison(tmp_path, candles, seed=7)
    assert (tmp_path / rel).exists()

    assert detect_overfit({"final_equity": 1500}, {"final_equity": 900}, threshold=0.3) is True
    assert detect_overfit({"final_equity": 1000}, {"final_equity": 950}, threshold=0.3) is False
