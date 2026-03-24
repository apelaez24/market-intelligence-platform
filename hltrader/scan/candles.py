"""Fetch 1h/4h returns for momentum acceleration scoring.

Data source priority:
1. HL snapshot fields (if present in snapshot dict)
2. HL candleSnapshot API (cached per scan cycle)

All fetches are cached per scan cycle to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# Module-level cache: coin -> {"ret_1h": float, "ret_4h": float, "ts": float}
_candle_cache: dict[str, dict] = {}
_cache_ttl: int = 300  # default 5 min, overridden by config


def set_cache_ttl(ttl: int) -> None:
    """Set the cache TTL (called from watcher on startup)."""
    global _cache_ttl
    _cache_ttl = ttl


def clear_cache() -> None:
    """Clear candle cache (useful between test runs)."""
    _candle_cache.clear()


def _fetch_candles(coin: str, interval: str = "1h", n: int = 5) -> list[dict] | None:
    """Fetch recent candles from HL candleSnapshot endpoint.

    Returns list of {"t": int, "o": str, "h": str, "l": str, "c": str, "v": str}
    or None on failure.
    """
    try:
        # HL expects startTime in ms; we go back n*interval_ms from now
        interval_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
        ims = interval_ms.get(interval, 3_600_000)
        start_ms = int((time.time() - n * ims / 1000) * 1000)

        resp = requests.post(
            HL_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval, "startTime": start_ms},
            },
            timeout=8,
        )
        resp.raise_for_status()
        candles = resp.json()
        if isinstance(candles, list) and candles:
            return candles
        return None
    except Exception as e:
        log.debug("candle fetch %s/%s failed: %s", coin, interval, e)
        return None


def _compute_returns(candles: list[dict]) -> dict:
    """Compute 1h and 4h returns from 1h candle data.

    Expects candles sorted by time ascending.
    Returns {"ret_1h": float|None, "ret_4h": float|None}.
    """
    if not candles or len(candles) < 2:
        return {"ret_1h": None, "ret_4h": None}

    latest_close = float(candles[-1]["c"])

    # 1h return: latest vs previous candle
    prev_close = float(candles[-2]["c"])
    ret_1h = ((latest_close / prev_close) - 1) * 100 if prev_close > 0 else None

    # 4h return: latest vs 4 candles back (if available)
    ret_4h = None
    if len(candles) >= 5:
        close_4h_ago = float(candles[-5]["c"])
        if close_4h_ago > 0:
            ret_4h = ((latest_close / close_4h_ago) - 1) * 100

    return {"ret_1h": ret_1h, "ret_4h": ret_4h}


def get_returns(coin: str) -> dict:
    """Get 1h/4h returns for a coin, using cache if fresh.

    Returns {"ret_1h": float|None, "ret_4h": float|None}.
    """
    now = time.time()
    cached = _candle_cache.get(coin)
    if cached and (now - cached.get("ts", 0)) < _cache_ttl:
        return {"ret_1h": cached.get("ret_1h"), "ret_4h": cached.get("ret_4h")}

    # Fetch 5 x 1h candles (enough for 4h lookback)
    candles = _fetch_candles(coin, interval="1h", n=5)
    result = _compute_returns(candles) if candles else {"ret_1h": None, "ret_4h": None}
    _candle_cache[coin] = {**result, "ts": now}
    return result


def build_returns_cache(coins: list[str]) -> dict[str, dict]:
    """Build returns cache for a batch of coins.

    Called once per scan cycle. Returns {coin: {"ret_1h": ..., "ret_4h": ...}}.
    """
    cache = {}
    for coin in coins:
        cache[coin] = get_returns(coin)
    return cache
