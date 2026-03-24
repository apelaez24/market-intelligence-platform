"""Pump-Fade Scoring Engine — mean-reversion indicators for extreme pump alerts.

Computes RSI-14 + Bollinger Bands to assess whether a pump is fading,
providing actionable "fade confidence" as an addendum to EXTREME PUMP alerts.

Feature-gated: HL_PUMP_FADE_ENABLED (default False).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process cache (symbol+timeframe -> closes list, with TTL)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, list[float]]] = {}


def _cache_get(key: str, ttl: int) -> list[float] | None:
    """Return cached closes if still fresh, else None."""
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < ttl:
            return data
    return None


def _cache_set(key: str, data: list[float]) -> None:
    _cache[key] = (time.time(), data)


def cache_clear() -> None:
    """Clear the in-process cache (useful for testing)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Indicator functions (pure math, no I/O)
# ---------------------------------------------------------------------------

def rsi_14(closes: list[float]) -> float | None:
    """Wilder-smoothed RSI-14.  Needs >= 15 closes.  Returns None if insufficient."""
    if len(closes) < 15:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    # Seed with SMA of first 14 deltas
    gains = [max(d, 0.0) for d in deltas[:14]]
    losses = [abs(min(d, 0.0)) for d in deltas[:14]]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    # Wilder smoothing for remaining deltas
    for d in deltas[14:]:
        avg_gain = (avg_gain * 13 + max(d, 0.0)) / 14
        avg_loss = (avg_loss * 13 + abs(min(d, 0.0))) / 14
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(closes: list[float]) -> list[float]:
    """Full RSI-14 series for rollover detection.

    Returns list of RSI values, one per close starting at index 14.
    Length = len(closes) - 14.  Returns empty list if < 15 closes.
    """
    if len(closes) < 15:
        return []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains_init = [max(d, 0.0) for d in deltas[:14]]
    losses_init = [abs(min(d, 0.0)) for d in deltas[:14]]
    avg_gain = sum(gains_init) / 14
    avg_loss = sum(losses_init) / 14

    result: list[float] = []
    # First RSI value (at index 14)
    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - (100.0 / (1.0 + rs)))

    for d in deltas[14:]:
        avg_gain = (avg_gain * 13 + max(d, 0.0)) / 14
        avg_loss = (avg_loss * 13 + abs(min(d, 0.0))) / 14
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))
    return result


def is_rsi_rollover(
    rsi_vals: list[float],
    overbought: float = 75.0,
    consecutive: int = 2,
    lookback: int = 6,
) -> bool:
    """Detect RSI rollover: was overbought recently AND now falling.

    - Look back ``lookback`` values in the RSI series.
    - Require at least one value > ``overbought`` in that window.
    - Require the last ``consecutive`` RSI values are each lower than the prior.
    """
    if len(rsi_vals) < max(lookback, consecutive + 1):
        return False
    window = rsi_vals[-lookback:]
    was_overbought = any(v > overbought for v in window)
    if not was_overbought:
        return False
    tail = rsi_vals[-(consecutive + 1):]
    for i in range(1, len(tail)):
        if tail[i] >= tail[i - 1]:
            return False
    return True


def bollinger_position(
    closes: list[float],
    length: int = 20,
    std: float = 2.0,
) -> dict[str, float] | None:
    """Bollinger Bands position.  Needs >= ``length`` closes.

    Returns {upper, middle, lower, pct_b, bandwidth} or None.
    pct_b: 0 = lower band, 1 = upper band, >1 = above upper.
    """
    if len(closes) < length:
        return None
    window = closes[-length:]
    middle = sum(window) / length
    variance = sum((x - middle) ** 2 for x in window) / length
    sd = variance ** 0.5
    upper = middle + std * sd
    lower = middle - std * sd
    band_range = upper - lower
    pct_b = (closes[-1] - lower) / band_range if band_range > 0 else 0.5
    bandwidth = band_range / middle if middle > 0 else 0.0
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    }


def bollinger_squeeze(bandwidths: list[float], threshold_pct: float = 20.0) -> bool:
    """True if current bandwidth is in the bottom ``threshold_pct`` percentile."""
    if len(bandwidths) < 2:
        return False
    current = bandwidths[-1]
    sorted_bw = sorted(bandwidths)
    rank = sorted_bw.index(current)
    percentile = (rank / (len(sorted_bw) - 1)) * 100
    return percentile <= threshold_pct


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

# Coinbase DB table lookup
_COINBASE_TABLES: dict[str, str] = {
    "BTC": "btcusd",
    "ETH": "ethusd",
    "SOL": "solusd",
    "TAO": "taousd",
}

# Timeframe -> table suffix
_TIMEFRAME_SUFFIX: dict[str, str] = {
    "5m": "_5m",
    "1h": "_1h",
}


def _get_closes_coinbase(
    symbol: str,
    timeframe: str,
    limit: int,
    pg_dsn: str,
) -> list[float] | None:
    """Fetch closes from coinbase_data DB.  Short-lived connection."""
    base = _COINBASE_TABLES.get(symbol)
    suffix = _TIMEFRAME_SUFFIX.get(timeframe)
    if not base or not suffix:
        return None
    table = base + suffix
    try:
        import psycopg2
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT close FROM {table} ORDER BY datetime DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        # Rows are DESC, reverse to chronological order
        return [float(r[0]) for r in reversed(rows)]
    except Exception as exc:
        log.debug("coinbase fetch failed for %s/%s: %s", symbol, timeframe, exc)
        return None


def _get_closes_hl(
    symbol: str,
    interval: str,
    limit: int,
    timeout: int = 10,
) -> list[float] | None:
    """Fetch closes from Hyperliquid candle API."""
    # HL API interval mapping
    hl_interval_map = {"5m": "5m", "1h": "1h"}
    hl_interval = hl_interval_map.get(interval)
    if not hl_interval:
        return None
    # HL wants startTime in ms; fetch enough history
    now_ms = int(time.time() * 1000)
    interval_ms = {"5m": 300_000, "1h": 3_600_000}[interval]
    start_ms = now_ms - (limit * interval_ms)
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": hl_interval,
                    "startTime": start_ms,
                    "endTime": now_ms,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        candles = resp.json()
        if not candles:
            return None
        closes = [float(c["c"]) for c in candles[-limit:]]
        return closes if len(closes) >= 15 else None
    except Exception as exc:
        log.debug("HL candle fetch failed for %s/%s: %s", symbol, interval, exc)
        return None


def get_closes(
    symbol: str,
    timeframe: str,
    limit: int,
    pg_dsn: str | None = None,
    cache_ttl: int = 900,
    hl_timeout: int = 10,
) -> list[float] | None:
    """Get close prices: Coinbase DB first (BTC/ETH/SOL/TAO), HL API fallback.

    Returns cached data if fresh.  Returns None if both sources fail.
    """
    cache_key = f"{symbol}:{timeframe}:{limit}"
    cached = _cache_get(cache_key, cache_ttl)
    if cached is not None:
        return cached

    closes = None
    # Try Coinbase DB first for supported symbols
    if pg_dsn and symbol in _COINBASE_TABLES:
        closes = _get_closes_coinbase(symbol, timeframe, limit, pg_dsn)

    # Fallback to HL API
    if closes is None:
        closes = _get_closes_hl(symbol, timeframe, limit, timeout=hl_timeout)

    if closes is not None:
        _cache_set(cache_key, closes)
    return closes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pump_fade(
    symbol: str,
    context: dict[str, Any],
    pg_dsn: str | None = None,
    cache_ttl: int = 900,
    hl_timeout: int = 10,
) -> dict[str, Any]:
    """Compute fade score for a pump candidate.

    context: {pct_24h, composite, funding_rate} from ShortCandidate.

    Returns: {
        score: 0-100,
        confidence: 0.0-1.0,
        rollover: bool,
        reasons: list[str],
        risk_flags: list[str],
        rsi_14_5m: float | None,
        rsi_14_1h: float | None,
        bb_pct_b: float | None,
        bb_bandwidth: float | None,
    }
    """
    result: dict[str, Any] = {
        "score": 0,
        "confidence": 0.0,
        "rollover": False,
        "reasons": [],
        "risk_flags": [],
        "rsi_14_5m": None,
        "rsi_14_1h": None,
        "bb_pct_b": None,
        "bb_bandwidth": None,
    }

    points = 0
    reasons: list[str] = []
    risk_flags: list[str] = []
    has_5m = False
    has_1h = False

    # --- 5m data ---
    closes_5m = get_closes(symbol, "5m", 40, pg_dsn=pg_dsn,
                           cache_ttl=cache_ttl, hl_timeout=hl_timeout)
    if closes_5m and len(closes_5m) >= 15:
        has_5m = True

        # RSI-14 (5m)
        rsi_val = rsi_14(closes_5m)
        if rsi_val is not None:
            result["rsi_14_5m"] = rsi_val
            if rsi_val > 80:
                points += 35
                reasons.append(f"RSI 5m extremely overbought ({rsi_val:.0f})")
            elif rsi_val > 70:
                points += 25
                reasons.append(f"RSI 5m overbought ({rsi_val:.0f})")

            # RSI rollover (5m)
            rsi_vals = rsi_series(closes_5m)
            if rsi_vals:
                rollover = is_rsi_rollover(rsi_vals)
                result["rollover"] = rollover
                if rollover:
                    points += 25
                    reasons.append("RSI rollover confirmed")
                elif rsi_val > 70:
                    risk_flags.append("RSI still climbing \u2014 no rollover yet")

        # Bollinger Bands (5m)
        if len(closes_5m) >= 20:
            bb = bollinger_position(closes_5m)
            if bb:
                result["bb_pct_b"] = bb["pct_b"]
                result["bb_bandwidth"] = bb["bandwidth"]
                if bb["pct_b"] > 1.0:
                    points += 15
                    reasons.append(f"Price above upper BB (pct_b={bb['pct_b']:.2f})")

                # Squeeze detection (need historical bandwidths)
                bw_series: list[float] = []
                for i in range(20, len(closes_5m) + 1):
                    sub_bb = bollinger_position(closes_5m[:i])
                    if sub_bb:
                        bw_series.append(sub_bb["bandwidth"])
                if bw_series and bollinger_squeeze(bw_series):
                    points -= 10
                    risk_flags.append("BB squeeze \u2014 breakout risk")

    # --- 1h data ---
    closes_1h = get_closes(symbol, "1h", 20, pg_dsn=pg_dsn,
                           cache_ttl=cache_ttl, hl_timeout=hl_timeout)
    if closes_1h and len(closes_1h) >= 15:
        has_1h = True
        rsi_1h = rsi_14(closes_1h)
        if rsi_1h is not None:
            result["rsi_14_1h"] = rsi_1h
            if rsi_1h > 70:
                points += 10
                reasons.append(f"RSI 1h confirms overbought ({rsi_1h:.0f})")

    # --- Funding from context ---
    funding = context.get("funding_rate", 0.0)
    if funding and funding > 0:
        points += 5
        reasons.append(f"Positive funding ({funding:.4%})")

    # --- Confidence ---
    if has_5m and has_1h:
        confidence = 1.0
    elif has_5m or has_1h:
        confidence = 0.6
    else:
        confidence = 0.0

    # --- Assemble result ---
    result["score"] = max(min(points, 100), 0)
    result["confidence"] = confidence
    result["reasons"] = reasons
    result["risk_flags"] = risk_flags
    return result


# ---------------------------------------------------------------------------
# Phase 2 stub
# ---------------------------------------------------------------------------

def get_regime_context_cached() -> dict:
    """Phase 2: DeFiLlama TVL + Polymarket macro flags. Stub for now."""
    return {"risk_regime": "unknown", "notes": "not implemented"}
