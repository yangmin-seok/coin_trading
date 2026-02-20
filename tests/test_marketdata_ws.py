from __future__ import annotations

import asyncio

from src.coin_trading.execution.marketdata import GapFiller, MarketDataWS, MemoryStateStore


class FakeRestClient:
    async def fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        out = []
        ts = start_ms
        while ts <= end_ms:
            out.append(
                {
                    "open_time": ts,
                    "close_time": ts + 299_999,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "num_trades": 1,
                }
            )
            ts += 300_000
        return out


def _closed_msg(open_ms: int) -> dict:
    return {
        "k": {
            "t": open_ms,
            "T": open_ms + 299_999,
            "o": "100",
            "h": "101",
            "l": "99",
            "c": "100.5",
            "v": "10",
            "n": 1,
            "x": True,
        }
    }


def _open_msg(open_ms: int) -> dict:
    msg = _closed_msg(open_ms)
    msg["k"]["x"] = False
    return msg


def test_marketdata_ws_ignores_unclosed_kline():
    async def _run():
        queue: asyncio.Queue = asyncio.Queue()
        ws = MarketDataWS("BTCUSDT", "5m", queue, MemoryStateStore())
        await ws.handle_kline_message(_open_msg(0))
        assert queue.empty()

    asyncio.run(_run())


def test_marketdata_ws_gapfill_and_order():
    async def _run():
        queue: asyncio.Queue = asyncio.Queue()
        state = MemoryStateStore()
        gapfiller = GapFiller("BTCUSDT", "5m", FakeRestClient())
        ws = MarketDataWS("BTCUSDT", "5m", queue, state, gap_filler=gapfiller)

        await ws.handle_kline_message(_closed_msg(0))
        await ws.handle_kline_message(_closed_msg(900_000))

        events = []
        while not queue.empty():
            events.append((await queue.get()).open_time_ms)
        assert events == [0, 300_000, 600_000, 900_000]

    asyncio.run(_run())
