"""Phase 3 + 6.5: Alert outcome tracking and evaluation.

Records every alert at send time, then evaluates 1h/4h/24h returns
and MFE/MAE using coinbase_data candles (primary) or HyperLiquid
candleSnapshot API (fallback for non-Coinbase symbols).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import psycopg2
import requests

log = logging.getLogger(__name__)

# Symbols with local 1h candle data in coinbase_data
_COINBASE_SYMBOLS: dict[str, str] = {
    "BTC": "btcusd_1h",
    "ETH": "ethusd_1h",
    "SOL": "solusd_1h",
    "TAO": "taousd_1h",
}

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# Per-run candle cache: (symbol, start_ms) -> list[dict]
# Bounded to MAX_CACHE_ENTRIES to protect Pi memory.
_hl_candle_cache: dict[tuple[str, int], list[dict]] = {}
MAX_CACHE_ENTRIES = 50

# Tolerance: accept candle within this many minutes of target time
CANDLE_TOLERANCE_MINUTES = 30

# Rate-limit: minimum seconds between HL API calls
_HL_MIN_INTERVAL_S = 0.5
_last_hl_call_ts: float = 0.0


def _clear_hl_cache() -> None:
    """Clear the per-run HL candle cache."""
    _hl_candle_cache.clear()


def _fetch_hl_candles(
    coin: str, interval: str, start_ms: int, n: int = 30
) -> list[dict] | None:
    """Fetch candles from HL candleSnapshot API with rate limiting.

    Returns list of {"t": int, "o": str, "h": str, "l": str, "c": str, "v": str}
    sorted by time ascending, or None on failure.
    """
    global _last_hl_call_ts

    cache_key = (coin, start_ms)
    if cache_key in _hl_candle_cache:
        return _hl_candle_cache[cache_key]

    # Evict oldest entries if cache is full
    if len(_hl_candle_cache) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_hl_candle_cache))
        del _hl_candle_cache[oldest_key]

    # Rate limit
    now = time.time()
    elapsed = now - _last_hl_call_ts
    if elapsed < _HL_MIN_INTERVAL_S:
        time.sleep(_HL_MIN_INTERVAL_S - elapsed)

    try:
        resp = requests.post(
            HL_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval, "startTime": start_ms},
            },
            timeout=10,
        )
        _last_hl_call_ts = time.time()
        resp.raise_for_status()
        candles = resp.json()
        if isinstance(candles, list) and candles:
            # Sort by timestamp ascending
            candles.sort(key=lambda c: c["t"])
            _hl_candle_cache[cache_key] = candles
            return candles
        return None
    except Exception as e:
        _last_hl_call_ts = time.time()
        log.debug("HL candle fetch %s/%s failed: %s", coin, interval, e)
        return None


def _find_candle_close_at(
    candles: list[dict], target_ms: int, tolerance_ms: int
) -> float | None:
    """Find the close price of the candle nearest to target_ms within tolerance.

    Candles must be sorted by time ascending.
    """
    best = None
    best_dist = tolerance_ms + 1
    for c in candles:
        dist = abs(c["t"] - target_ms)
        if dist < best_dist:
            best_dist = dist
            best = float(c["c"])
    if best_dist <= tolerance_ms:
        return best
    return None


def _hl_lookup_return(
    coin: str, alert_ts: datetime, price_at_alert: float, *, hours: int
) -> float | None:
    """Look up return % at alert_ts + hours using HL 1h candles.

    Fetches a window of candles covering [alert_ts, alert_ts + hours + tolerance].
    Tolerance: 30 minutes.
    """
    alert_ms = int(alert_ts.timestamp() * 1000)
    # Fetch from alert time, need enough candles to cover the horizon
    candles = _fetch_hl_candles(coin, "1h", alert_ms, n=hours + 2)
    if not candles:
        return None

    target_ms = alert_ms + hours * 3_600_000
    tolerance_ms = CANDLE_TOLERANCE_MINUTES * 60_000

    close_price = _find_candle_close_at(candles, target_ms, tolerance_ms)
    if close_price is not None and price_at_alert > 0:
        return (close_price - price_at_alert) / price_at_alert * 100.0
    return None


def _hl_lookup_mfe_mae(
    coin: str, alert_ts: datetime, price_at_alert: float
) -> tuple[float | None, float | None]:
    """Look up MFE/MAE over 24h using HL 1h candles.

    MFE = max favorable excursion (highest high vs alert price)
    MAE = max adverse excursion (lowest low vs alert price)
    Returns signed percentages.
    """
    alert_ms = int(alert_ts.timestamp() * 1000)
    candles = _fetch_hl_candles(coin, "1h", alert_ms, n=26)
    if not candles:
        return None, None

    end_ms = alert_ms + 24 * 3_600_000
    max_high = None
    min_low = None

    for c in candles:
        if c["t"] <= alert_ms or c["t"] > end_ms:
            continue
        h = float(c["h"])
        lo = float(c["l"])
        if max_high is None or h > max_high:
            max_high = h
        if min_low is None or lo < min_low:
            min_low = lo

    if max_high is not None and min_low is not None and price_at_alert > 0:
        mfe = (max_high - price_at_alert) / price_at_alert * 100.0
        mae = (min_low - price_at_alert) / price_at_alert * 100.0
        return mfe, mae
    return None, None


def record_alert(
    pg_dsn: str,
    *,
    symbol: str,
    alert_type: str,
    score: float,
    pump_score: float,
    funding_score: float,
    oi_score: float,
    accel_score: float,
    liquidity_score: float,
    price_at_alert: float,
    alert_timestamp: datetime,
    conviction_score: float | None = None,
    cluster_type: str | None = None,
) -> bool:
    """Insert a new alert into alert_outcomes. Fail-open: never raises."""
    try:
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO alert_outcomes
                       (symbol, alert_type, score, pump_score, funding_score,
                        oi_score, accel_score, liquidity_score,
                        price_at_alert, alert_timestamp, conviction_score,
                        cluster_type)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        symbol, alert_type, score, pump_score, funding_score,
                        oi_score, accel_score, liquidity_score,
                        price_at_alert, alert_timestamp, conviction_score,
                        cluster_type,
                    ),
                )
            conn.commit()
            log.info("alert_outcome recorded: %s %s score=%.1f price=%.4f",
                     symbol, alert_type, score, price_at_alert)
            return True
        finally:
            conn.close()
    except Exception as exc:
        log.warning("record_alert failed (non-fatal): %s", exc)
        return False


def evaluate_pending(
    pg_dsn: str,
    pg_dsn_coinbase: str,
    *,
    batch_size: int = 100,
) -> dict:
    """Evaluate pending alert outcomes.

    Primary: coinbase_data candles (BTC, ETH, SOL, TAO).
    Fallback: HyperLiquid candleSnapshot API (all other symbols).

    Returns {"evaluated_1h": N, "evaluated_4h": N, "evaluated_24h": N,
             "skipped": N, "hl_evaluated": N}.
    """
    summary = {
        "evaluated_1h": 0, "evaluated_4h": 0, "evaluated_24h": 0,
        "skipped": 0, "hl_evaluated": 0,
    }

    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.error("evaluate_pending: cannot connect to market_intel: %s", exc)
        return summary

    cb_conn = None
    try:
        cb_conn = psycopg2.connect(pg_dsn_coinbase)
    except Exception as exc:
        log.warning("evaluate_pending: cannot connect to coinbase_data: %s "
                    "(will use HL fallback for all symbols)", exc)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, symbol, price_at_alert, alert_timestamp,
                          evaluated_1h, evaluated_4h, evaluated_24h
                   FROM alert_outcomes
                   WHERE NOT evaluated_1h OR NOT evaluated_4h OR NOT evaluated_24h
                   ORDER BY alert_timestamp ASC
                   LIMIT %s""",
                (batch_size,),
            )
            rows = cur.fetchall()

        now = datetime.now(timezone.utc)

        # Group by symbol to reuse HL candle cache
        for row in rows:
            row_id, symbol, price_at_alert, alert_ts, ev_1h, ev_4h, ev_24h = row
            table = _COINBASE_SYMBOLS.get(symbol)
            use_coinbase = table is not None and cb_conn is not None

            age_hours = (now - alert_ts).total_seconds() / 3600.0

            if use_coinbase:
                source = "coinbase"
            else:
                source = "hyperliquid"

            # 1h evaluation
            if not ev_1h and age_hours >= 1.5:
                if use_coinbase:
                    ret = _lookup_return(cb_conn, table, alert_ts, price_at_alert, hours=1)
                else:
                    ret = _hl_lookup_return(symbol, alert_ts, price_at_alert, hours=1)
                _update_eval(conn, row_id, "1h", ret)
                summary["evaluated_1h"] += 1
                if not use_coinbase:
                    summary["hl_evaluated"] += 1
                log.info("eval %s %s 1h ret=%s source=%s",
                         symbol, alert_ts.isoformat(), ret, source)

            # 4h evaluation
            if not ev_4h and age_hours >= 4.5:
                if use_coinbase:
                    ret = _lookup_return(cb_conn, table, alert_ts, price_at_alert, hours=4)
                else:
                    ret = _hl_lookup_return(symbol, alert_ts, price_at_alert, hours=4)
                _update_eval(conn, row_id, "4h", ret)
                summary["evaluated_4h"] += 1
                if not use_coinbase:
                    summary["hl_evaluated"] += 1
                log.info("eval %s %s 4h ret=%s source=%s",
                         symbol, alert_ts.isoformat(), ret, source)

            # 24h evaluation (includes MFE/MAE)
            if not ev_24h and age_hours >= 24.5:
                if use_coinbase:
                    ret = _lookup_return(cb_conn, table, alert_ts, price_at_alert, hours=24)
                    mfe, mae = _lookup_mfe_mae(cb_conn, table, alert_ts, price_at_alert)
                else:
                    ret = _hl_lookup_return(symbol, alert_ts, price_at_alert, hours=24)
                    mfe, mae = _hl_lookup_mfe_mae(symbol, alert_ts, price_at_alert)
                _update_eval_24h(conn, row_id, ret, mfe, mae)
                summary["evaluated_24h"] += 1
                if not use_coinbase:
                    summary["hl_evaluated"] += 1
                log.info("eval %s %s 24h ret=%s mfe=%s mae=%s source=%s",
                         symbol, alert_ts.isoformat(), ret, mfe, mae, source)

    except Exception as exc:
        log.error("evaluate_pending error: %s", exc)
    finally:
        _clear_hl_cache()
        if cb_conn:
            cb_conn.close()
        conn.close()

    log.info("evaluate_pending: %s", summary)
    return summary


def reset_null_evals(pg_dsn: str) -> dict:
    """Reset evaluated flags where return values are NULL.

    This repairs rows that were incorrectly marked as evaluated
    (e.g., non-Coinbase symbols before HL fallback was available).

    Returns {"reset_1h": N, "reset_4h": N, "reset_24h": N}.
    """
    result = {"reset_1h": 0, "reset_4h": 0, "reset_24h": 0}

    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.error("reset_null_evals: cannot connect: %s", exc)
        return result

    try:
        with conn.cursor() as cur:
            # Reset 1h
            cur.execute(
                """UPDATE alert_outcomes
                   SET evaluated_1h = FALSE
                   WHERE evaluated_1h = TRUE AND eval_1h_return IS NULL"""
            )
            result["reset_1h"] = cur.rowcount

            # Reset 4h
            cur.execute(
                """UPDATE alert_outcomes
                   SET evaluated_4h = FALSE
                   WHERE evaluated_4h = TRUE AND eval_4h_return IS NULL"""
            )
            result["reset_4h"] = cur.rowcount

            # Reset 24h
            cur.execute(
                """UPDATE alert_outcomes
                   SET evaluated_24h = FALSE
                   WHERE evaluated_24h = TRUE
                     AND eval_24h_return IS NULL
                     AND mfe_24h IS NULL
                     AND mae_24h IS NULL"""
            )
            result["reset_24h"] = cur.rowcount

        conn.commit()
        log.info("reset_null_evals: %s", result)
    except Exception as exc:
        log.error("reset_null_evals error: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()

    return result


def _lookup_return(
    cb_conn, table: str, alert_ts: datetime, price_at_alert: float, *, hours: int
) -> float | None:
    """Look up return % at alert_timestamp + hours from coinbase 1h candles."""
    try:
        with cb_conn.cursor() as cur:
            cur.execute(
                f"SELECT close FROM {table}"
                f" WHERE datetime >= %s + interval '{hours} hours'"
                f" AND datetime <= %s + interval '{hours} hours 30 minutes'"
                " ORDER BY datetime ASC LIMIT 1",
                (alert_ts, alert_ts),
            )
            row = cur.fetchone()
            if row and row[0] and price_at_alert:
                return (float(row[0]) - price_at_alert) / price_at_alert * 100.0
    except Exception as exc:
        log.warning("_lookup_return failed (%s, %dh): %s", table, hours, exc)
    return None


def _lookup_mfe_mae(
    cb_conn, table: str, alert_ts: datetime, price_at_alert: float
) -> tuple[float | None, float | None]:
    """Look up MFE (max high) and MAE (min low) over 24h window."""
    try:
        with cb_conn.cursor() as cur:
            cur.execute(
                f"SELECT MAX(high), MIN(low) FROM {table}"
                " WHERE datetime > %s AND datetime <= %s + interval '24 hours'",
                (alert_ts, alert_ts),
            )
            row = cur.fetchone()
            if row and row[0] is not None and row[1] is not None and price_at_alert:
                mfe = (float(row[0]) - price_at_alert) / price_at_alert * 100.0
                mae = (float(row[1]) - price_at_alert) / price_at_alert * 100.0
                return mfe, mae
    except Exception as exc:
        log.warning("_lookup_mfe_mae failed (%s): %s", table, exc)
    return None, None


def _mark_non_coinbase(conn, row_id, ev_1h: bool, ev_4h: bool, ev_24h: bool) -> None:
    """Mark non-coinbase symbol as evaluated with NULL returns.

    DEPRECATED: Kept for backward compat but no longer called by evaluate_pending.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE alert_outcomes
                   SET evaluated_1h = TRUE, evaluated_4h = TRUE, evaluated_24h = TRUE
                   WHERE id = %s""",
                (row_id,),
            )
        conn.commit()
    except Exception as exc:
        log.warning("_mark_non_coinbase failed: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass


def _update_eval(conn, row_id, horizon: str, return_pct: float | None) -> None:
    """Update a single evaluation horizon."""
    col_ret = f"eval_{horizon}_return"
    col_flag = f"evaluated_{horizon}"
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE alert_outcomes"
                f" SET {col_ret} = %s, {col_flag} = TRUE"
                " WHERE id = %s",
                (return_pct, row_id),
            )
        conn.commit()
    except Exception as exc:
        log.warning("_update_eval failed (%s): %s", horizon, exc)
        try:
            conn.rollback()
        except Exception:
            pass


def _update_eval_24h(
    conn, row_id, return_pct: float | None, mfe: float | None, mae: float | None
) -> None:
    """Update 24h evaluation with return + MFE/MAE."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE alert_outcomes
                   SET eval_24h_return = %s, mfe_24h = %s, mae_24h = %s,
                       evaluated_24h = TRUE
                   WHERE id = %s""",
                (return_pct, mfe, mae, row_id),
            )
        conn.commit()
    except Exception as exc:
        log.warning("_update_eval_24h failed: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass


def get_weekly_stats(pg_dsn: str) -> dict:
    """Aggregate alert performance stats over the last 7 days."""
    result = {
        "total_alerts": 0,
        "by_type": [],
        "top_symbols": [],
        "worst_symbols": [],
    }

    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.error("get_weekly_stats: cannot connect: %s", exc)
        return result

    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                       alert_type,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE evaluated_1h AND eval_1h_return < 0) AS wins_1h,
                       COUNT(*) FILTER (WHERE evaluated_1h AND eval_1h_return IS NOT NULL) AS eval_1h,
                       COUNT(*) FILTER (WHERE evaluated_4h AND eval_4h_return < 0) AS wins_4h,
                       COUNT(*) FILTER (WHERE evaluated_4h AND eval_4h_return IS NOT NULL) AS eval_4h,
                       COUNT(*) FILTER (WHERE evaluated_24h AND eval_24h_return < 0) AS wins_24h,
                       COUNT(*) FILTER (WHERE evaluated_24h AND eval_24h_return IS NOT NULL) AS eval_24h,
                       AVG(eval_1h_return) FILTER (WHERE evaluated_1h AND eval_1h_return IS NOT NULL) AS avg_1h,
                       AVG(eval_24h_return) FILTER (WHERE evaluated_24h AND eval_24h_return IS NOT NULL) AS avg_24h
                   FROM alert_outcomes
                   WHERE alert_timestamp >= now() - interval '7 days'
                   GROUP BY alert_type"""
            )
            rows = cur.fetchall()

            total = 0
            by_type = []
            for r in rows:
                (atype, cnt, w1, e1, w4, e4, w24, e24, avg1, avg24) = r
                total += cnt
                by_type.append({
                    "alert_type": atype,
                    "total": cnt,
                    "wins_1h": w1, "eval_1h": e1,
                    "wins_4h": w4, "eval_4h": e4,
                    "wins_24h": w24, "eval_24h": e24,
                    "avg_1h": round(float(avg1), 2) if avg1 is not None else None,
                    "avg_24h": round(float(avg24), 2) if avg24 is not None else None,
                })
            result["total_alerts"] = total
            result["by_type"] = by_type

            cur.execute(
                """SELECT symbol, AVG(eval_24h_return) AS avg_ret, COUNT(*) AS n
                   FROM alert_outcomes
                   WHERE evaluated_24h AND eval_24h_return IS NOT NULL
                     AND alert_timestamp >= now() - interval '7 days'
                   GROUP BY symbol HAVING COUNT(*) >= 2
                   ORDER BY avg_ret ASC LIMIT 5"""
            )
            result["top_symbols"] = [
                {"symbol": r[0], "avg_ret": round(float(r[1]), 2), "n": r[2]}
                for r in cur.fetchall()
            ]

            cur.execute(
                """SELECT symbol, AVG(eval_24h_return) AS avg_ret, COUNT(*) AS n
                   FROM alert_outcomes
                   WHERE evaluated_24h AND eval_24h_return IS NOT NULL
                     AND alert_timestamp >= now() - interval '7 days'
                   GROUP BY symbol HAVING COUNT(*) >= 2
                   ORDER BY avg_ret DESC LIMIT 5"""
            )
            result["worst_symbols"] = [
                {"symbol": r[0], "avg_ret": round(float(r[1]), 2), "n": r[2]}
                for r in cur.fetchall()
            ]

    except Exception as exc:
        log.error("get_weekly_stats error: %s", exc)
    finally:
        conn.close()

    return result
