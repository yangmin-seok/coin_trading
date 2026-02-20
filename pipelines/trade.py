from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.baselines import BuyAndHold
from config.loader import load_config
from execution.marketdata import CandleClosedEvent, GapFiller, MarketDataWS, MemoryStateStore
from execution.orders import OrderExecutor, OrderRequest
from execution.reconcile import Reconciler
from execution.risk import RiskLimits, RiskManager
from execution.state import PortfolioState
from integrations.binance_rest import BinanceRESTClient
from monitoring.alerts import Alert, AlertEngine
from monitoring.metrics import MetricsLogger


@dataclass(slots=True)
class TradeRuntime:
    market_ws: MarketDataWS
    reconciler: Reconciler
    rest_client: BinanceRESTClient
    queue: asyncio.Queue
    metrics: MetricsLogger
    alerts: AlertEngine
    risk: RiskManager
    orders: OrderExecutor
    state: PortfolioState
    policy: BuyAndHold


def build_runtime() -> TradeRuntime:
    cfg = load_config()
    queue: asyncio.Queue = asyncio.Queue()

    rest_client = BinanceRESTClient()
    state_store = MemoryStateStore()
    gap_filler = GapFiller(cfg.symbol, cfg.interval, rest_client=rest_client)
    market_ws = MarketDataWS(cfg.symbol, cfg.interval, queue=queue, state_store=state_store, gap_filler=gap_filler)
    reconciler = Reconciler()
    metrics = MetricsLogger(path=(Path("runs") / "runtime_metrics.jsonl"))
    alerts = AlertEngine()
    risk = RiskManager(RiskLimits(max_position_ratio=1.0, min_position_ratio=0.0, max_notional_per_trade=5_000.0))
    orders = OrderExecutor()
    state = PortfolioState(cash=10_000.0, position_qty=0.0, equity=10_000.0, peak_equity=10_000.0)
    policy = BuyAndHold()
    return TradeRuntime(
        market_ws=market_ws,
        reconciler=reconciler,
        rest_client=rest_client,
        queue=queue,
        metrics=metrics,
        alerts=alerts,
        risk=risk,
        orders=orders,
        state=state,
        policy=policy,
    )


def process_candle_event(runtime: TradeRuntime, event: CandleClosedEvent) -> dict[str, Any]:
    price = float(event.c)
    runtime.state.mark_to_market(price)
    equity = max(runtime.state.equity, 1e-12)
    current_ratio = (runtime.state.position_qty * price) / equity

    target_ratio = float(runtime.policy.act(obs=None, info={"close": price}))
    decision = runtime.risk.evaluate_target(target_ratio, current_ratio, equity, price)
    if not decision.approved:
        runtime.metrics.emit({"decision": "rejected", "reasons": decision.reasons})
        return {"status": "risk_rejected", "reasons": decision.reasons}

    side = "BUY" if decision.order_qty > 0 else "SELL"
    req = OrderRequest(symbol=event.symbol, side=side, quantity=abs(decision.order_qty))
    result = runtime.orders.place_market_order(req, mark_price=price)
    if not result.accepted:
        runtime.metrics.incr("order_failures", 1)
        runtime.metrics.emit({"decision": "order_failed", "reason": result.reason})
        return {"status": "order_failed", "reason": result.reason}

    signed_qty = result.executed_qty if side == "BUY" else -result.executed_qty
    runtime.state.cash -= signed_qty * result.avg_price
    runtime.state.position_qty += signed_qty
    runtime.state.mark_to_market(price)
    runtime.metrics.emit({"decision": "filled", "target_ratio": decision.target_position_ratio, "qty": signed_qty})
    return {"status": "filled", "qty": signed_qty, "price": result.avg_price}


def reconcile_once(runtime: TradeRuntime, internal_total: float, quote_asset: str = "USDT") -> Alert | None:
    account_payload = runtime.rest_client.get_account()
    exchange_total = runtime.reconciler.extract_spot_total(account_payload, quote_asset=quote_asset)
    result = runtime.reconciler.compare_total_balance(internal_total, exchange_total)
    if not result.matched:
        runtime.metrics.incr("reconcile_mismatches", 1)
    runtime.metrics.emit({"reconcile_reason": result.reason})
    return runtime.alerts.check_reconcile(result.matched, result.reason)


def run() -> str:
    runtime = build_runtime()
    return f"trade runtime ready: symbol={runtime.market_ws.symbol}, interval={runtime.market_ws.interval}"
