from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.coin_trading.pipelines.train_flow import orchestrator


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


@pytest.fixture
def patched_meta(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orchestrator, "write_meta", lambda run_dir: (run_dir / "meta.json").write_text("{}", encoding="utf-8"))


@pytest.fixture
def patched_train_sb3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestrator,
        "train_sb3",
        lambda *_args, **_kwargs: {"enabled": False, "reason": "insufficient_split_rows"},
    )


@pytest.fixture
def fixed_run_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(orchestrator, "make_run_id", lambda *args, **kwargs: "smoke_train_run")
    return "smoke_train_run"
