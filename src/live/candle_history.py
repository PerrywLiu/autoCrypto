"""Validation shared by durable strategy checkpoints and shadow inputs."""
import math


def validate_history(candles, interval_ms, now_ms=None):
    if not isinstance(candles, list) or not candles:
        raise ValueError("indicator_history_empty")
    previous = None
    for row in candles:
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError("indicator_history_row")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
            raise ValueError("indicator_history_number")
        stamp, opening, high, low, close, volume = row
        if not isinstance(stamp, int) or stamp < 0 or stamp % interval_ms:
            raise ValueError("indicator_history_timestamp")
        if min(opening, high, low, close) <= 0 or volume < 0 or low > min(opening, close) or high < max(opening, close) or high < low:
            raise ValueError("indicator_history_ohlcv")
        if previous is not None and stamp - previous != interval_ms:
            raise ValueError("indicator_history_gap")
        if now_ms is not None and stamp + interval_ms > now_ms:
            raise ValueError("indicator_history_unclosed")
        previous = stamp


def checkpoint_candles(state, interval_ms, now_ms=None):
    checkpoint = state.get("indicator_history")
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise ValueError("indicator_history_version")
    candles = checkpoint.get("candles")
    validate_history(candles, interval_ms, now_ms)
    return candles
