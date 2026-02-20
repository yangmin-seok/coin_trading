from __future__ import annotations

import asyncio

from execution.marketdata import CandleClosedEvent
from pipelines.trade import build_runtime, run_forever


class FakeStream:
    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.05)
        return None

    async def close(self):
        self.closed = True


class DummyRESTClient:
    def get_account(self):
        return {"balances": [{"asset": "USDT", "free": "10000", "locked": "0"}]}


def _kline_msg(open_time: int):
    return {
        "k": {
            "t": open_time,
            "T": open_time + 299_999,
            "o": "100",
            "h": "101",
            "l": "99",
            "c": "100",
            "v": "10",
            "n": 1,
            "x": True,
        }
    }


def test_run_forever_processes_queue_and_stops(tmp_path):
    async def _run():
        runtime = build_runtime()
        runtime.metrics.path = tmp_path / "runtime_metrics.jsonl"
        runtime.rest_client = DummyRESTClient()
        stream = FakeStream([_kline_msg(0)])
        stop_event = asyncio.Event()

        async def stopper():
            await asyncio.sleep(0.2)
            stop_event.set()

        task = asyncio.create_task(run_forever(runtime, stream, stop_event=stop_event, reconcile_interval_s=0.05))
        await stopper()
        await task

        assert stream.closed
        assert runtime.state.position_qty > 0

    asyncio.run(_run())
