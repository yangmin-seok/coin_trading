from __future__ import annotations

from execution.reconcile import Reconciler


def test_reconcile_matches_within_threshold():
    r = Reconciler(abs_threshold=1.0, rel_threshold=0.01)
    out = r.compare_total_balance(1000.0, 1000.5)
    assert out.matched


def test_extract_spot_total_quote_asset_only():
    r = Reconciler()
    payload = {
        "balances": [
            {"asset": "USDT", "free": "10", "locked": "2"},
            {"asset": "BTC", "free": "0.1", "locked": "0"},
        ]
    }
    assert r.extract_spot_total(payload, quote_asset="USDT") == 12.0
