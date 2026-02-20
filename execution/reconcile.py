from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReconcileResult:
    matched: bool
    balance_diff: float
    reason: str


class Reconciler:
    def __init__(self, abs_threshold: float = 1e-6, rel_threshold: float = 1e-4) -> None:
        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    def compare_total_balance(self, internal_total: float, exchange_total: float) -> ReconcileResult:
        diff = abs(internal_total - exchange_total)
        rel = diff / max(abs(exchange_total), 1e-12)
        matched = not (diff > self.abs_threshold and rel > self.rel_threshold)
        reason = "ok" if matched else f"balance_mismatch(diff={diff:.8f}, rel={rel:.8f})"
        return ReconcileResult(matched=matched, balance_diff=diff, reason=reason)

    def extract_spot_total(self, account_payload: dict[str, Any], quote_asset: str = "USDT") -> float:
        total = 0.0
        for row in account_payload.get("balances", []):
            asset = row.get("asset")
            free = float(row.get("free", 0.0))
            locked = float(row.get("locked", 0.0))
            if asset == quote_asset:
                total += free + locked
        return total
