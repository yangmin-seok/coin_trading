from __future__ import annotations

import asyncio

from src.coin_trading.integrations.binance_ws_user import BinanceUserWS, extract_user_payload
from src.coin_trading.integrations.telegram import TelegramSender


def test_extract_user_payload_handles_combined_stream():
    payload = extract_user_payload({"stream": "x", "data": {"e": "executionReport", "i": 1}})
    assert payload["e"] == "executionReport"


def test_user_ws_routes_event_to_queue():
    async def _run():
        q = asyncio.Queue()
        ws = BinanceUserWS(queue=q)
        out = await ws.handle_message({"e": "executionReport", "i": 2})
        assert out.event_type == "executionReport"
        queued = await q.get()
        assert queued.payload["i"] == 2

    asyncio.run(_run())


def test_telegram_sender_builds_api_url():
    sender = TelegramSender(bot_token="abc", chat_id="123")
    assert sender.api_url.endswith("/botabc/sendMessage")
