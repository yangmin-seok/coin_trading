from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from src.coin_trading.agents.baselines import VolTarget
from src.coin_trading.config.loader import load_config
from src.coin_trading.execution.marketdata import GapFiller, MarketDataWS, MemoryStateStore
from src.coin_trading.execution.orders import OrderManager
from src.coin_trading.execution.reconcile import Reconciler
from src.coin_trading.execution.risk import RiskManager
from src.coin_trading.execution.state import PortfolioState
from src.coin_trading.features.online import OnlineFeatureEngine
from src.coin_trading.integrations.binance_rest import BinanceRESTClient
from src.coin_trading.monitoring.alerts import Alert, AlertEngine
from src.coin_trading.monitoring.metrics import MetricsLogger


@dataclass(slots=True)
class TradeRuntime:
    market_ws: MarketDataWS
    reconciler: Reconciler
    rest_client: BinanceRESTClient
    queue: asyncio.Queue
    metrics: MetricsLogger
    alerts: AlertEngine
    features: OnlineFeatureEngine
    risk: RiskManager
    orders: OrderManager
    policy: VolTarget
    state: PortfolioState


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
    features = OnlineFeatureEngine()
    risk = RiskManager()
    orders = OrderManager(client=rest_client)
    policy = VolTarget()
    state = PortfolioState(cash=1_000.0, position_qty=0.0, equity=1_000.0, peak_equity=1_000.0)
    return TradeRuntime(
        market_ws=market_ws,
        reconciler=reconciler,
        rest_client=rest_client,
        queue=queue,
        metrics=metrics,
        alerts=alerts,
        features=features,
        risk=risk,
        orders=orders,
        policy=policy,
        state=state,
    )


def reconcile_once(runtime: TradeRuntime, internal_total: float, quote_asset: str = "USDT") -> Alert | None:
    account_payload = runtime.rest_client.get_account()
    exchange_total = runtime.reconciler.extract_spot_total(account_payload, quote_asset=quote_asset)
    result = runtime.reconciler.compare_total_balance(internal_total, exchange_total)
    if not result.matched:
        runtime.metrics.incr("reconcile_mismatches", 1)
    runtime.metrics.emit({"reconcile_reason": result.reason})
    return runtime.alerts.check_reconcile(result.matched, result.reason)


def process_market_event(runtime: TradeRuntime, event) -> dict:
    runtime.state.mark_to_market(event.c)
    obs = runtime.features.update({"close": event.c, "high": event.h, "low": event.l, "volume": event.v})
    target = runtime.policy.act(obs=None, info={"close": event.c, "logret_1": obs.get("logret_1", 0.0)})
    decision = runtime.risk.approve_target(
        target,
        equity=runtime.state.equity,
        peak_equity=runtime.state.peak_equity,
        notional=runtime.state.position_qty * event.c,
    )

    order_intent = None
    if decision.approved:
        current_position = (runtime.state.position_qty * event.c) / runtime.state.equity if runtime.state.equity > 0 else 0.0
        order_intent = runtime.orders.target_to_intent(
            runtime.market_ws.symbol,
            current_position=current_position,
            target_position=decision.target,
            equity=runtime.state.equity,
            price=event.c,
        )

    runtime.metrics.incr("market_events", 1)
    runtime.metrics.emit(
        {
            "event_ts": event.open_time_ms,
            "price": event.c,
            "equity": runtime.state.equity,
            "target": decision.target,
            "risk_reason": decision.reason,
            "order_side": order_intent.side if order_intent else None,
            "order_qty": order_intent.quantity if order_intent else 0.0,
        }
    )
    return {
        "target": decision.target,
        "approved": decision.approved,
        "order_intent": order_intent,
        "equity": runtime.state.equity,
    }


def run(max_events: int = 0, timeout_s: float = 0.0) -> str:
    runtime = build_runtime()
    if max_events <= 0:
        return f"trade runtime ready: symbol={runtime.market_ws.symbol}, interval={runtime.market_ws.interval}"

    async def _consume() -> int:
        processed = 0
        while processed < max_events:
            try:
                event = await asyncio.wait_for(runtime.queue.get(), timeout=timeout_s if timeout_s > 0 else None)
            except TimeoutError:
                break
            process_market_event(runtime, event)
            processed += 1
        return processed

    processed = asyncio.run(_consume())
    return (
        f"trade runtime ready: symbol={runtime.market_ws.symbol}, "
        f"interval={runtime.market_ws.interval}, processed_events={processed}"
    )
