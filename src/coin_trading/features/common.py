from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class FeatureState:
    closes: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    highs: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    lows: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    volumes: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    logrets: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    ema_fast: float | None = None
    ema_slow: float | None = None
    ema_signal: float | None = None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return float("nan")
    diffs = [closes[i] - closes[i - 1] for i in range(len(closes) - period, len(closes))]
    gains = [max(d, 0.0) for d in diffs]
    losses = [abs(min(d, 0.0)) for d in diffs]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(prev: float | None, value: float, span: int) -> float:
    alpha = 2 / (span + 1)
    return value if prev is None else alpha * value + (1 - alpha) * prev


def update_features(state: FeatureState, candle: dict[str, float]) -> dict[str, float]:
    close = float(candle["close"])
    high = float(candle["high"])
    low = float(candle["low"])
    volume = float(candle["volume"])

    prev_close = state.closes[-1] if state.closes else None
    logret_1 = math.log(close / prev_close) if prev_close else float("nan")

    state.closes.append(close)
    state.highs.append(high)
    state.lows.append(low)
    state.volumes.append(volume)
    if not math.isnan(logret_1):
        state.logrets.append(logret_1)

    ret_window = list(state.logrets)[-20:]
    hl = (high - low) / close if close else 0.0
    hl_window = [((h - l) / c) for h, l, c in zip(list(state.highs)[-20:], list(state.lows)[-20:], list(state.closes)[-20:]) if c]

    vol_window = list(state.volumes)[-20:]
    vol_mean = _mean(vol_window) if len(vol_window) >= 2 else float("nan")
    vol_std = _std(vol_window) if len(vol_window) >= 2 else float("nan")

    state.ema_fast = _ema(state.ema_fast, close, 12)
    state.ema_slow = _ema(state.ema_slow, close, 26)
    macd = state.ema_fast - state.ema_slow if (state.ema_fast is not None and state.ema_slow is not None) else float("nan")
    state.ema_signal = _ema(state.ema_signal, macd, 9) if not math.isnan(macd) else state.ema_signal
    macd_signal = state.ema_signal if state.ema_signal is not None else float("nan")

    close_window = list(state.closes)[-20:]
    bb_mid = _mean(close_window) if len(close_window) >= 2 else float("nan")
    bb_std = _std(close_window) if len(close_window) >= 2 else float("nan")

    return {
        "logret_1": logret_1,
        "roll_mean_logret_20": _mean(ret_window) if len(ret_window) >= 2 else float("nan"),
        "roll_std_logret_20": _std(ret_window) if len(ret_window) >= 2 else float("nan"),
        "hl_range": hl,
        "roll_mean_hl_range_20": _mean(hl_window) if len(hl_window) >= 2 else float("nan"),
        "roll_std_hl_range_20": _std(hl_window) if len(hl_window) >= 2 else float("nan"),
        "vol_z_20": ((volume - vol_mean) / vol_std) if len(vol_window) >= 2 and vol_std > 0 else float("nan"),
        "rsi_14": _rsi(list(state.closes), 14),
        "macd_12_26": macd,
        "macd_signal_9": macd_signal,
        "bb_mid_20": bb_mid,
        "bb_upper_20": bb_mid + 2 * bb_std if not math.isnan(bb_mid) and not math.isnan(bb_std) else float("nan"),
        "bb_lower_20": bb_mid - 2 * bb_std if not math.isnan(bb_mid) and not math.isnan(bb_std) else float("nan"),
    }
