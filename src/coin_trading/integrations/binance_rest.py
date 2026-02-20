from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True)
class BinanceRESTClient:
    base_url: str = "https://api.binance.com/api"
    api_key: str | None = None
    api_secret: str | None = None
    timeout_s: float = 10.0
    max_retries: int = 3
    backoff_s: float = 0.5

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = params or {}
        headers = {"User-Agent": "coin-trading/0.1"}

        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("api_key/api_secret are required for signed endpoints")
            params["timestamp"] = int(time.time() * 1000)
            query = urlencode(params)
            signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.api_key
        elif self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        query = urlencode(params)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = Request(url=url, method=method, headers=headers)
                with urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
                err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.backoff_s * (2**attempt)
                time.sleep(sleep_s)
        raise RuntimeError(f"binance request failed: {method} {path}") from err

    async def fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._request(
            "GET",
            "/v3/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": int(limit),
            },
            signed=False,
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                    "quote_volume": float(row[7]),
                    "num_trades": int(row[8]),
                    "taker_buy_base_volume": float(row[9]),
                    "taker_buy_quote_volume": float(row[10]),
                    "ignore": float(row[11]),
                }
            )
        return out

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/v3/account", signed=True)

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/v3/openOrders", params=params, signed=True)
