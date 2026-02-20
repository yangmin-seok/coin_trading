from __future__ import annotations

from src.coin_trading.execution.orders import OrderManager
from src.coin_trading.execution.risk import RiskManager
from src.coin_trading.execution.state import PortfolioState


class DummyClient:
    def __init__(self):
        self.calls = []

    def _request(self, method, path, params=None, signed=False):
        self.calls.append((method, path, params, signed))
        return {"ok": True, "params": params}


def test_order_manager_builds_market_order_request():
    client = DummyClient()
    om = OrderManager(client=client)
    out = om.create_market_order("btcusdt", "buy", 0.01)
    assert out["ok"]
    method, path, params, signed = client.calls[-1]
    assert (method, path, signed) == ("POST", "/v3/order", True)
    assert params["symbol"] == "BTCUSDT"


def test_order_manager_apply_execution_report_updates_state():
    state = PortfolioState(cash=1000.0, position_qty=0.0, equity=1000.0, peak_equity=1000.0)
    om = OrderManager(client=DummyClient())
    om.apply_execution_report(state, {"X": "FILLED", "S": "BUY", "l": "1.0", "L": "100.0", "n": "0.1"})
    assert state.position_qty == 1.0
    assert state.cash == 899.9


def test_risk_manager_rejects_on_drawdown_limit():
    rm = RiskManager(max_drawdown=0.2)
    decision = rm.approve_target(0.8, equity=70.0, peak_equity=100.0, notional=10.0)
    assert not decision.approved
    assert decision.reason == "drawdown_limit"
