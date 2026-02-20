from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PortfolioState:
    cash: float
    position_qty: float
    equity: float
    peak_equity: float

    def mark_to_market(self, price: float) -> None:
        self.equity = self.cash + self.position_qty * price
        self.peak_equity = max(self.peak_equity, self.equity)
