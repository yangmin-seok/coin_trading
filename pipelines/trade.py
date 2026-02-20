from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from config.loader import load_config
from execution.marketdata import GapFiller, MarketDataWS, MemoryStateStore
from execution.reconcile import Reconciler
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
    return TradeRuntime(
        market_ws=market_ws,
        reconciler=reconciler,
        rest_client=rest_client,
        queue=queue,
        metrics=metrics,
        alerts=alerts,
    )


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
