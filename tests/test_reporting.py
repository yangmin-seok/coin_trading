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
from src.coin_trading.report.plotting import write_learning_curve_artifacts


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


def test_learning_curve_svg_has_secondary_axis_and_legacy_paths(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    plots_dir = tmp_path / "plots"
    history = [
        {
            "timesteps": 1000,
            "val": {
                "sharpe": 0.7,
                "max_drawdown": 0.1,
                "final_equity": 1100.0,
                "pnl": 100.0,
                "turnover": 0.15,
                "total_cost": 2.0,
                "cost_pnl_ratio": 0.02,
            },
            "train": {
                "loss": 0.11,
                "entropy_loss": 0.02,
                "value_loss": 0.34,
                "loss_collected": True,
            },
        },
        {
            "timesteps": 2000,
            "val": {
                "sharpe": 0.9,
                "max_drawdown": 0.08,
                "final_equity": 1150.0,
                "pnl": 150.0,
                "turnover": 0.2,
                "total_cost": 2.8,
                "cost_pnl_ratio": 0.018,
            },
            "train": {
                "loss": None,
                "entropy_loss": None,
                "value_loss": None,
                "loss_collected": False,
            },
        },
    ]

    write_learning_curve_artifacts(history, reports_dir, plots_dir)

    csv_path = reports_dir / "learning_curve.csv"
    json_path = reports_dir / "learning_curve.json"
    svg_path = plots_dir / "learning_curve.svg"
    summary_svg_path = plots_dir / "learning_curve_summary.svg"
    assert csv_path.exists()
    assert json_path.exists()
    assert svg_path.exists()
    assert summary_svg_path.exists()

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == [
        "timesteps",
        "val_sharpe",
        "val_max_drawdown",
        "val_final_equity",
        "loss",
        "entropy_loss",
        "value_loss",
        "loss_collected",
    ]
    assert frame.loc[0, "loss"] == 0.11
    assert frame.loc[0, "entropy_loss"] == 0.02
    assert frame.loc[0, "value_loss"] == 0.34
    assert bool(frame.loc[0, "loss_collected"]) is True
    assert pd.isna(frame.loc[1, "loss"])
    assert pd.isna(frame.loc[1, "entropy_loss"])
    assert pd.isna(frame.loc[1, "value_loss"])
    assert bool(frame.loc[1, "loss_collected"]) is False

    json_text = json_path.read_text(encoding="utf-8")
    assert '"loss": null' in json_text

    schema_path = reports_dir / "learning_curve.schema.json"
    assert schema_path.exists()
    assert '"version": 2' in schema_path.read_text(encoding="utf-8")

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "value [R]" in svg_text
    assert "[R] val_turnover" in svg_text
    assert "[L] val_return_pct" in svg_text

    summary_svg_text = summary_svg_path.read_text(encoding="utf-8")
    assert "Validation Summary" in summary_svg_text
    assert "Final Equity" in summary_svg_text


def test_learning_curve_legacy_loss_keys_are_supported(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    plots_dir = tmp_path / "plots"
    history = [
        {
            "timesteps": 500,
            "val": {"sharpe": 0.1, "max_drawdown": 0.05, "final_equity": 1010.0},
            "loss": 0.0,
            "entropy_loss": 0.0,
            "value_loss": 0.0,
        }
    ]

    write_learning_curve_artifacts(history, reports_dir, plots_dir)
    frame = pd.read_csv(reports_dir / "learning_curve.csv")
    assert frame.loc[0, "loss"] == 0.0
    assert frame.loc[0, "entropy_loss"] == 0.0
    assert frame.loc[0, "value_loss"] == 0.0
    assert bool(frame.loc[0, "loss_collected"]) is True
