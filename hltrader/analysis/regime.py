"""Phase 5: Market Regime Engine.

6-state regime classifier combining BTC trend, volatility state (ATR),
breadth metrics (from scan snapshots), and risk classification.
Stores snapshots to DB, injects into alerts, optionally adjusts conviction.

Regime codes: {UPTREND,DOWNTREND,CHOP}_{EXPANSION,CONTRACTION}
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import psycopg2

log = logging.getLogger(__name__)

# ── Cache ────────────────────────────────────────────────────

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> object | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object, ttl: int) -> None:
    _cache[key] = (time.time() + ttl, value)


def cache_clear() -> None:
    """Clear regime cache (for testing)."""
    _cache.clear()


# ── Hysteresis state ─────────────────────────────────────────

_last_vol_state: str = "contraction"


def _reset_vol_state() -> None:
    """Reset hysteresis state (for testing)."""
    global _last_vol_state
    _last_vol_state = "contraction"


# ── Result dataclass ─────────────────────────────────────────

@dataclass
class RegimeResult:
    regime_code: str       # UPTREND_EXPANSION, etc.
    btc_trend: str         # up|down|chop
    vol_state: str         # expansion|contraction
    risk_state: str        # risk_on|risk_off|neutral
    metrics: dict          # all numeric metrics
    timestamp: datetime
    cached: bool = False


# ── EMA helper (reused from conviction.py pattern) ───────────

def _compute_ema(values: list[float], period: int) -> list[float]:
    """Compute EMA series. Returns list of length len(values) - period + 1."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(values[:period]) / period]  # SMA seed
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


# ── BTC Trend ────────────────────────────────────────────────

def _classify_btc_trend(
    pg_dsn_coinbase: str,
    slope_up: float = 0.002,
    slope_down: float = -0.002,
) -> tuple[str, dict]:
    """Classify BTC trend from 1h candles aggregated to 4h.

    Returns (trend_label, metrics_dict).
    trend_label: "up", "down", or "chop".
    """
    metrics: dict = {}
    try:
        conn = psycopg2.connect(pg_dsn_coinbase)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT close FROM btcusd_1h
                       ORDER BY datetime DESC LIMIT 200"""
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 80:
            return "chop", metrics

        # Reverse to chronological, cast Decimal->float
        closes_1h = [float(r[0]) for r in reversed(rows)]

        # Aggregate to 4h: take every 4th close
        closes_4h = [closes_1h[i] for i in range(3, len(closes_1h), 4)]

        if len(closes_4h) < 50:
            return "chop", metrics

        # EMA20 and EMA50 on 4h closes
        ema20 = _compute_ema(closes_4h, 20)
        ema50 = _compute_ema(closes_4h, 50)

        if not ema20 or not ema50:
            return "chop", metrics

        metrics["btc_ema20"] = ema20[-1]
        metrics["btc_ema50"] = ema50[-1]

        # slope_4h = pct change of EMA20 over last 6 points
        if len(ema20) >= 6:
            recent = ema20[-6:]
            slope = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0.0
            metrics["btc_slope_4h"] = slope
        else:
            metrics["btc_slope_4h"] = 0.0
            return "chop", metrics

        # Classification
        ema20_last = ema20[-1]
        ema50_last = ema50[-1]
        slope = metrics["btc_slope_4h"]

        if ema20_last > ema50_last and slope > slope_up:
            return "up", metrics
        elif ema20_last < ema50_last and slope < slope_down:
            return "down", metrics
        else:
            return "chop", metrics

    except Exception as exc:
        log.warning("_classify_btc_trend failed (non-fatal): %s", exc)
        return "chop", metrics


# ── Volatility State ─────────────────────────────────────────

def _classify_vol_state(
    pg_dsn_coinbase: str,
    atr_expand: float = 1.25,
    atr_contract: float = 0.85,
) -> tuple[str, dict]:
    """Classify volatility state using ATR(14) on 4h close-to-close.

    Returns (vol_label, metrics_dict).
    vol_label: "expansion" or "contraction".
    Uses hysteresis: if ratio is between thresholds, keeps previous state.
    """
    global _last_vol_state
    metrics: dict = {}

    try:
        conn = psycopg2.connect(pg_dsn_coinbase)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT close FROM btcusd_1h
                       ORDER BY datetime DESC LIMIT 200"""
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 80:
            return _last_vol_state, metrics

        closes_1h = [float(r[0]) for r in reversed(rows)]

        # Aggregate to 4h
        closes_4h = [closes_1h[i] for i in range(3, len(closes_1h), 4)]

        if len(closes_4h) < 30:  # need 14 for ATR + some history for median
            return _last_vol_state, metrics

        # True Range (close-to-close): |close[i] - close[i-1]|
        tr_values = [abs(closes_4h[i] - closes_4h[i - 1]) for i in range(1, len(closes_4h))]

        if len(tr_values) < 14:
            return _last_vol_state, metrics

        # Wilder-smoothed ATR(14): start with SMA, then exponential smoothing
        atr = sum(tr_values[:14]) / 14.0
        for tr in tr_values[14:]:
            atr = (atr * 13 + tr) / 14.0

        metrics["atr_4h"] = atr

        # Median ATR over last 50 TR values
        recent_tr = tr_values[-50:] if len(tr_values) >= 50 else tr_values
        median_atr = statistics.median(recent_tr)

        if median_atr == 0:
            return _last_vol_state, metrics

        ratio = atr / median_atr
        metrics["atr_ratio"] = ratio

        if ratio >= atr_expand:
            _last_vol_state = "expansion"
        elif ratio <= atr_contract:
            _last_vol_state = "contraction"
        # else: hysteresis — keep previous state

        return _last_vol_state, metrics

    except Exception as exc:
        log.warning("_classify_vol_state failed (non-fatal): %s", exc)
        return _last_vol_state, metrics


# ── Breadth ──────────────────────────────────────────────────

def _compute_breadth(snapshots: list[dict]) -> dict:
    """Compute breadth metrics from scan snapshots.

    Returns dict with:
      breadth_pump10_pct: % coins with 24h% >= +10%
      breadth_dump10_pct: % coins with 24h% <= -10%
      funding_median: median funding rate
      funding_pctl: % with positive funding
    """
    if not snapshots:
        return {
            "breadth_pump10_pct": 0.0,
            "breadth_dump10_pct": 0.0,
            "funding_median": 0.0,
            "funding_pctl": 0.0,
        }

    total = len(snapshots)
    pump_count = sum(1 for s in snapshots if s.get("pct_24h", 0) >= 10.0)
    dump_count = sum(1 for s in snapshots if s.get("pct_24h", 0) <= -10.0)

    funding_rates = [s.get("funding_rate", 0.0) for s in snapshots]
    positive_funding = sum(1 for f in funding_rates if f > 0)

    return {
        "breadth_pump10_pct": pump_count / total,
        "breadth_dump10_pct": dump_count / total,
        "funding_median": statistics.median(funding_rates) if funding_rates else 0.0,
        "funding_pctl": positive_funding / total,
    }


# ── Risk State ───────────────────────────────────────────────

def _classify_risk_state(
    btc_trend: str,
    vol_state: str,
    breadth: dict,
    breadth_pump_threshold: float = 0.18,
    breadth_dump_threshold: float = 0.18,
) -> str:
    """Classify risk state.

    risk_off: (down AND expansion) OR dump_breadth >= threshold
    risk_on: (up AND expansion) OR pump_breadth >= threshold
    neutral: otherwise
    risk_off takes precedence if both conditions true.
    """
    dump_pct = breadth.get("breadth_dump10_pct", 0.0)
    pump_pct = breadth.get("breadth_pump10_pct", 0.0)

    is_risk_off = (
        (btc_trend == "down" and vol_state == "expansion")
        or dump_pct >= breadth_dump_threshold
    )
    is_risk_on = (
        (btc_trend == "up" and vol_state == "expansion")
        or pump_pct >= breadth_pump_threshold
    )

    # risk_off takes precedence
    if is_risk_off:
        return "risk_off"
    if is_risk_on:
        return "risk_on"
    return "neutral"


# ── Main Entry Point ─────────────────────────────────────────

def compute_regime(
    snapshots: list[dict],
    *,
    pg_dsn_coinbase: str = "",
    slope_up: float = 0.002,
    slope_down: float = -0.002,
    atr_expand: float = 1.25,
    atr_contract: float = 0.85,
    breadth_pump_threshold: float = 0.18,
    breadth_dump_threshold: float = 0.18,
    cache_ttl: int = 600,
) -> RegimeResult:
    """Compute market regime from BTC data + scan snapshots.

    Cached for cache_ttl seconds. Returns RegimeResult.
    """
    cache_key = "market_regime"
    cached = _cache_get(cache_key)
    if cached is not None:
        cached.cached = True
        return cached

    now = datetime.now(timezone.utc)

    # BTC trend
    if pg_dsn_coinbase:
        btc_trend, trend_metrics = _classify_btc_trend(
            pg_dsn_coinbase, slope_up=slope_up, slope_down=slope_down
        )
    else:
        btc_trend, trend_metrics = "chop", {}

    # Volatility
    if pg_dsn_coinbase:
        vol_state, vol_metrics = _classify_vol_state(
            pg_dsn_coinbase, atr_expand=atr_expand, atr_contract=atr_contract
        )
    else:
        vol_state, vol_metrics = "contraction", {}

    # Breadth
    breadth = _compute_breadth(snapshots)

    # Risk state
    risk_state = _classify_risk_state(
        btc_trend, vol_state, breadth,
        breadth_pump_threshold=breadth_pump_threshold,
        breadth_dump_threshold=breadth_dump_threshold,
    )

    # Regime code
    trend_map = {"up": "UPTREND", "down": "DOWNTREND", "chop": "CHOP"}
    vol_map = {"expansion": "EXPANSION", "contraction": "CONTRACTION"}
    regime_code = f"{trend_map[btc_trend]}_{vol_map[vol_state]}"

    # Merge all metrics
    metrics = {**trend_metrics, **vol_metrics, **breadth}

    result = RegimeResult(
        regime_code=regime_code,
        btc_trend=btc_trend,
        vol_state=vol_state,
        risk_state=risk_state,
        metrics=metrics,
        timestamp=now,
    )

    _cache_set(cache_key, result, cache_ttl)
    return result


# ── DB Storage ───────────────────────────────────────────────

def store_regime_snapshot(pg_dsn: str, regime: RegimeResult) -> None:
    """Store regime snapshot to market_regime_snapshots table.

    Fail-open: DB errors are logged but never raised.
    """
    try:
        m = regime.metrics
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO market_regime_snapshots
                       (regime_code, btc_trend, vol_state, risk_state,
                        btc_ema20, btc_ema50, btc_slope_4h,
                        atr_4h, atr_ratio,
                        breadth_pump10_pct, breadth_dump10_pct,
                        funding_median, funding_pctl, snapshot_ts)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        regime.regime_code,
                        regime.btc_trend,
                        regime.vol_state,
                        regime.risk_state,
                        m.get("btc_ema20"),
                        m.get("btc_ema50"),
                        m.get("btc_slope_4h"),
                        m.get("atr_4h"),
                        m.get("atr_ratio"),
                        m.get("breadth_pump10_pct"),
                        m.get("breadth_dump10_pct"),
                        m.get("funding_median"),
                        m.get("funding_pctl"),
                        regime.timestamp,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        log.info("regime_snapshot stored: %s", regime.regime_code)
    except Exception as exc:
        log.warning("store_regime_snapshot failed (non-fatal): %s", exc)
