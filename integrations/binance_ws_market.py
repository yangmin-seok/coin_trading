from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BinanceMarketWSConfig:
    base_url: str = "wss://stream.binance.com:9443"


def extract_kline_payload(message: dict[str, Any]) -> dict[str, Any]:
    """Handle both raw `/ws` and combined `/stream` payload shapes."""
    if "data" in message and isinstance(message["data"], dict):
        return message["data"]
    return message
