from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Alert:
    level: str
    code: str
    message: str


class AlertEngine:
    def check_reconcile(self, matched: bool, reason: str) -> Alert | None:
        if matched:
            return None
        return Alert(level="critical", code="RECONCILE_MISMATCH", message=reason)

    def check_drawdown(self, drawdown: float, dd_limit: float) -> Alert | None:
        if drawdown >= dd_limit:
            return Alert(level="critical", code="DD_LIMIT", message=f"drawdown={drawdown:.4f} >= limit={dd_limit:.4f}")
        if drawdown >= dd_limit * 0.8:
            return Alert(level="warning", code="DD_NEAR_LIMIT", message=f"drawdown={drawdown:.4f}")
        return None
