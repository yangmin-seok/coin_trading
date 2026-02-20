from __future__ import annotations

from pathlib import Path

from monitoring.alerts import AlertEngine
from monitoring.metrics import MetricsLogger


def test_metrics_logger_writes_jsonl(tmp_path: Path):
    log = tmp_path / "m.jsonl"
    m = MetricsLogger(path=log)
    m.incr("ws_reconnects")
    m.emit({"tag": "unit"})
    assert log.exists()
    assert "ws_reconnects" in log.read_text(encoding="utf-8")


def test_alert_engine_reconcile_and_dd():
    engine = AlertEngine()
    a1 = engine.check_reconcile(False, "mismatch")
    assert a1 is not None and a1.code == "RECONCILE_MISMATCH"
    a2 = engine.check_drawdown(0.21, 0.2)
    assert a2 is not None and a2.code == "DD_LIMIT"
