from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class RuntimeCounters:
    ws_reconnects: int = 0
    gapfills: int = 0
    reconcile_mismatches: int = 0
    order_failures: int = 0
    market_events: int = 0


@dataclass(slots=True)
class MetricsLogger:
    path: Path
    counters: RuntimeCounters = field(default_factory=RuntimeCounters)

    def incr(self, field_name: str, value: int = 1) -> None:
        if not hasattr(self.counters, field_name):
            raise ValueError(f"unknown counter: {field_name}")
        current = getattr(self.counters, field_name)
        setattr(self.counters, field_name, current + value)

    def emit(self, extra: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ws_reconnects": self.counters.ws_reconnects,
            "gapfills": self.counters.gapfills,
            "reconcile_mismatches": self.counters.reconcile_mismatches,
            "order_failures": self.counters.order_failures,
            "market_events": self.counters.market_events,
        }
        if extra:
            payload.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
