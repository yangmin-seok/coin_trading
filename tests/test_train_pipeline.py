from __future__ import annotations

from pathlib import Path

from pipelines.train import run


def test_train_pipeline_writes_summary(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/default.yaml").write_text(
        """
mode: demo
exchange: binance
market: spot
symbol: BTCUSDT
interval: 5m
seed: 7
timezone_storage: UTC
report_timezone: Asia/Seoul
reward:
  type: log_return
  lambda_turnover: 0.001
  lambda_dd: 0.1
  dd_limit: 0.2
execution:
  fee_rate: 0.001
  slippage_bps: 2.0
  max_step_change: 0.25
  min_delta: 0.01
features:
  version: v1
  windows:
    return: 20
    rsi: 14
    bb: 20
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
split:
  train: ["2022-01-01", "2024-12-31"]
  val: ["2025-01-01", "2025-06-30"]
  test: ["2025-07-01", "2025-12-31"]
""".strip(),
        encoding="utf-8",
    )

    run_id = run()
    out = tmp_path / "runs" / run_id
    assert (out / "meta.json").exists()
    assert (out / "train_summary.json").exists()
