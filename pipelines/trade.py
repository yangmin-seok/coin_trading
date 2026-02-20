from __future__ import annotations

import asyncio
from dataclasses import dataclass

from config.loader import load_config
from execution.marketdata import GapFiller, MarketDataWS, MemoryStateStore
from execution.reconcile import Reconciler
from integrations.binance_rest import BinanceRESTClient


@dataclass(slots=True)
class TradeRuntime:
    market_ws: MarketDataWS
    reconciler: Reconciler
    rest_client: BinanceRESTClient
    queue: asyncio.Queue


def build_runtime() -> TradeRuntime:
    cfg = load_config()
    queue: asyncio.Queue = asyncio.Queue()

    rest_client = BinanceRESTClient()
    state_store = MemoryStateStore()
    gap_filler = GapFiller(cfg.symbol, cfg.interval, rest_client=rest_client)
    market_ws = MarketDataWS(cfg.symbol, cfg.interval, queue=queue, state_store=state_store, gap_filler=gap_filler)
    reconciler = Reconciler()
    return TradeRuntime(market_ws=market_ws, reconciler=reconciler, rest_client=rest_client, queue=queue)


def run() -> str:
    runtime = build_runtime()
    return f"trade runtime ready: symbol={runtime.market_ws.symbol}, interval={runtime.market_ws.interval}"
