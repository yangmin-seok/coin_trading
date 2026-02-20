from __future__ import annotations

from src.coin_trading.features.online import OnlineFeatureEngine
from src.coin_trading.features.parity_test import replay_and_compare


def test_features_offline_online_parity(sample_candles):
    engine = OnlineFeatureEngine()
    replay_and_compare(sample_candles, engine, n_steps=100, atol=1e-9)
