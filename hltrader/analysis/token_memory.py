"""Phase 7: Token Personality Memory.

Learns how each token tends to behave after alerts.
Stores compact aggregates only — no raw history.
Computed from alert_outcomes using bounded lookback window.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg2

log = logging.getLogger(__name__)

# Runtime read cache: symbol -> TokenPersonality
_memory_cache: dict[str, "TokenPersonality"] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 600  # 10 minutes


@dataclass
class TokenPersonality:
    symbol: str
    sample_size: int
    sample_size_clustered: int
    sample_size_unclustered: int
    win_1h: float | None
    win_4h: float | None
    win_24h: float | None
    avg_ret_1h: float | None
    avg_ret_4h: float | None
    avg_ret_24h: float | None
    avg_mfe_24h: float | None
    avg_mae_24h: float | None
    trend_follow_score: float
    mean_reversion_score: float
    reversal_speed_score: float
    cluster_sensitivity: float
    best_regime: str | None
    worst_regime: str | None
    confidence_score: float
    personality_label: str
    updated_at: datetime | None = None


def clear_cache() -> None:
    """Clear the read cache."""
    _memory_cache.clear()
    global _cache_ts
    _cache_ts = 0.0


def get_personality(pg_dsn: str, symbol: str) -> TokenPersonality | None:
    """Get token personality from cache or DB. Returns None if not found."""
    global _cache_ts
    now = time.time()

    # Refresh cache if stale
    if now - _cache_ts > _CACHE_TTL or not _memory_cache:
        _load_all_personalities(pg_dsn)
        _cache_ts = now

    return _memory_cache.get(symbol)


def _load_all_personalities(pg_dsn: str) -> None:
    """Load all token personalities into cache. Small table, single query."""
    _memory_cache.clear()
    try:
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT symbol, sample_size, sample_size_clustered,
                              sample_size_unclustered,
                              win_1h, win_4h, win_24h,
                              avg_ret_1h, avg_ret_4h, avg_ret_24h,
                              avg_mfe_24h, avg_mae_24h,
                              trend_follow_score, mean_reversion_score,
                              reversal_speed_score, cluster_sensitivity,
                              best_regime, worst_regime,
                              confidence_score, personality_label, updated_at
                       FROM token_personality_memory"""
                )
                for row in cur.fetchall():
                    tp = TokenPersonality(
                        symbol=row[0], sample_size=row[1],
                        sample_size_clustered=row[2],
                        sample_size_unclustered=row[3],
                        win_1h=_f(row[4]), win_4h=_f(row[5]), win_24h=_f(row[6]),
                        avg_ret_1h=_f(row[7]), avg_ret_4h=_f(row[8]),
                        avg_ret_24h=_f(row[9]),
                        avg_mfe_24h=_f(row[10]), avg_mae_24h=_f(row[11]),
                        trend_follow_score=_f(row[12]) or 0.0,
                        mean_reversion_score=_f(row[13]) or 0.0,
                        reversal_speed_score=_f(row[14]) or 0.0,
                        cluster_sensitivity=_f(row[15]) or 0.0,
                        best_regime=row[16], worst_regime=row[17],
                        confidence_score=_f(row[18]) or 0.0,
                        personality_label=row[19] or "mixed",
                        updated_at=row[20],
                    )
                    _memory_cache[tp.symbol] = tp
        finally:
            conn.close()
    except Exception as exc:
        log.warning("_load_all_personalities failed: %s", exc)


def _f(val) -> float | None:
    """Safe float cast."""
    if val is None:
        return None
    return float(val)


# ── Computation Engine ────────────────────────────────────────


def compute_token_memory(
    pg_dsn: str,
    *,
    lookback_days: int = 30,
    min_sample: int = 5,
    min_sample_regime: int = 3,
) -> dict:
    """Recompute token personality memory from alert_outcomes.

    Returns {"computed": N, "skipped": N, "total_rows": N}.
    """
    summary = {"computed": 0, "skipped": 0, "total_rows": 0}

    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.error("compute_token_memory: cannot connect: %s", exc)
        return summary

    try:
        # Fetch all evaluated rows within lookback window
        with conn.cursor() as cur:
            cur.execute(
                """SELECT symbol, eval_1h_return, eval_4h_return, eval_24h_return,
                          mfe_24h, mae_24h, cluster_type
                   FROM alert_outcomes
                   WHERE eval_24h_return IS NOT NULL
                     AND alert_timestamp >= now() - interval '%s days'
                   ORDER BY symbol""",
                (lookback_days,),
            )
            all_rows = cur.fetchall()

        summary["total_rows"] = len(all_rows)

        # Group by symbol
        by_symbol: dict[str, list] = {}
        for row in all_rows:
            sym = row[0]
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(row)

        # Free the big list
        del all_rows

        # Fetch per-symbol per-regime data
        regime_data = _fetch_regime_stats(conn, lookback_days)

        # Compute and upsert each symbol
        for sym, rows in by_symbol.items():
            if len(rows) < min_sample:
                summary["skipped"] += 1
                continue

            tp = _compute_single(sym, rows, regime_data.get(sym, {}), min_sample_regime)
            _upsert_personality(conn, tp)
            summary["computed"] += 1

        # Clean up symbols no longer meeting threshold
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM token_personality_memory
                   WHERE symbol NOT IN (
                       SELECT symbol FROM alert_outcomes
                       WHERE eval_24h_return IS NOT NULL
                         AND alert_timestamp >= now() - interval '%s days'
                       GROUP BY symbol HAVING COUNT(*) >= %s
                   )""",
                (lookback_days, min_sample),
            )
            pruned = cur.rowcount
            if pruned:
                log.info("Pruned %d symbols below min_sample", pruned)
        conn.commit()

    except Exception as exc:
        log.error("compute_token_memory error: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()
        clear_cache()

    log.info("compute_token_memory: %s", summary)
    return summary


def _fetch_regime_stats(conn, lookback_days: int) -> dict:
    """Fetch per-symbol per-regime avg_24h return.

    Returns {symbol: {regime_code: {"avg_24h": float, "n": int}}}.
    """
    result: dict[str, dict] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT a.symbol, r.regime_code,
                          AVG(a.eval_24h_return) AS avg_24h,
                          COUNT(*) AS n
                   FROM alert_outcomes a
                   LEFT JOIN LATERAL (
                       SELECT regime_code
                       FROM market_regime_snapshots
                       WHERE snapshot_ts <= a.alert_timestamp
                       ORDER BY snapshot_ts DESC LIMIT 1
                   ) r ON true
                   WHERE a.eval_24h_return IS NOT NULL
                     AND a.alert_timestamp >= now() - interval '%s days'
                     AND r.regime_code IS NOT NULL
                   GROUP BY a.symbol, r.regime_code""",
                (lookback_days,),
            )
            for sym, regime, avg24, n in cur.fetchall():
                if sym not in result:
                    result[sym] = {}
                result[sym][regime] = {"avg_24h": float(avg24), "n": n}
    except Exception as exc:
        log.warning("_fetch_regime_stats failed: %s", exc)
    return result


def _compute_single(
    symbol: str,
    rows: list,
    regime_stats: dict,
    min_sample_regime: int,
) -> TokenPersonality:
    """Compute personality for a single symbol from its alert_outcomes rows."""
    n = len(rows)

    # Extract arrays
    ret_1h = [float(r[1]) for r in rows if r[1] is not None]
    ret_4h = [float(r[2]) for r in rows if r[2] is not None]
    ret_24h = [float(r[3]) for r in rows if r[3] is not None]
    mfe_24h = [float(r[4]) for r in rows if r[4] is not None]
    mae_24h = [float(r[5]) for r in rows if r[5] is not None]

    # Cluster split
    clustered = [r for r in rows if r[6] is not None]
    unclustered = [r for r in rows if r[6] is None]

    # Basic stats
    win_1h = sum(1 for x in ret_1h if x < 0) / len(ret_1h) if ret_1h else None
    win_4h = sum(1 for x in ret_4h if x < 0) / len(ret_4h) if ret_4h else None
    win_24h = sum(1 for x in ret_24h if x < 0) / len(ret_24h) if ret_24h else None

    avg_1h = sum(ret_1h) / len(ret_1h) if ret_1h else None
    avg_4h = sum(ret_4h) / len(ret_4h) if ret_4h else None
    avg_24h = sum(ret_24h) / len(ret_24h) if ret_24h else None
    avg_mfe = sum(mfe_24h) / len(mfe_24h) if mfe_24h else None
    avg_mae = sum(mae_24h) / len(mae_24h) if mae_24h else None

    # Trend-follow score (0-100)
    # Higher if returns tend to continue in the short direction (negative)
    # Weight: 4h contributes 40%, 24h contributes 60%
    trend_follow = _compute_trend_follow(ret_4h, ret_24h)

    # Mean-reversion score (0-100)
    # Higher if returns reverse after alert (negative returns = good for shorts)
    mean_reversion = _compute_mean_reversion(ret_1h, ret_4h, ret_24h)

    # Reversal speed score (0-100)
    # Higher if token fades quickly (1h negative while 24h may not be)
    reversal_speed = _compute_reversal_speed(ret_1h, ret_4h)

    # Cluster sensitivity
    cluster_sens = _compute_cluster_sensitivity(clustered, unclustered)

    # Best/worst regime
    best_regime, worst_regime = _compute_regime_extremes(
        regime_stats, min_sample_regime
    )

    # Confidence score (0-100)
    confidence = _compute_confidence(n, ret_24h)

    # Personality label
    label = _classify_personality(
        trend_follow, mean_reversion, reversal_speed, cluster_sens, confidence
    )

    return TokenPersonality(
        symbol=symbol,
        sample_size=n,
        sample_size_clustered=len(clustered),
        sample_size_unclustered=len(unclustered),
        win_1h=_round_or_none(win_1h, 4),
        win_4h=_round_or_none(win_4h, 4),
        win_24h=_round_or_none(win_24h, 4),
        avg_ret_1h=_round_or_none(avg_1h, 4),
        avg_ret_4h=_round_or_none(avg_4h, 4),
        avg_ret_24h=_round_or_none(avg_24h, 4),
        avg_mfe_24h=_round_or_none(avg_mfe, 4),
        avg_mae_24h=_round_or_none(avg_mae, 4),
        trend_follow_score=round(trend_follow, 2),
        mean_reversion_score=round(mean_reversion, 2),
        reversal_speed_score=round(reversal_speed, 2),
        cluster_sensitivity=round(cluster_sens, 4),
        best_regime=best_regime,
        worst_regime=worst_regime,
        confidence_score=round(confidence, 2),
        personality_label=label,
    )


def _compute_trend_follow(ret_4h: list[float], ret_24h: list[float]) -> float:
    """Trend-follow score: higher if returns continue in alert direction.

    For our short-biased alert universe, "continuation" means price keeps
    going up after the alert (bad for shorts = trend following).
    Score 0-100 where 100 = strong trend follower.
    """
    if not ret_4h or not ret_24h:
        return 50.0  # neutral

    # Fraction of positive returns (price kept going up = trend following)
    pos_4h = sum(1 for x in ret_4h if x > 0) / len(ret_4h)
    pos_24h = sum(1 for x in ret_24h if x > 0) / len(ret_24h)

    # Weight 24h more heavily
    raw = pos_4h * 0.4 + pos_24h * 0.6

    # Scale to 0-100
    return max(0.0, min(raw * 100.0, 100.0))


def _compute_mean_reversion(
    ret_1h: list[float], ret_4h: list[float], ret_24h: list[float]
) -> float:
    """Mean-reversion score: higher if returns reverse after alert.

    For shorts: negative returns = price dropped = mean reverted.
    Score 0-100 where 100 = strong mean reverter.
    """
    if not ret_24h:
        return 50.0

    # Fraction of negative returns
    neg_1h = sum(1 for x in ret_1h if x < 0) / len(ret_1h) if ret_1h else 0.5
    neg_4h = sum(1 for x in ret_4h if x < 0) / len(ret_4h) if ret_4h else 0.5
    neg_24h = sum(1 for x in ret_24h if x < 0) / len(ret_24h)

    # Weight longer horizons more
    raw = neg_1h * 0.2 + neg_4h * 0.3 + neg_24h * 0.5

    return max(0.0, min(raw * 100.0, 100.0))


def _compute_reversal_speed(ret_1h: list[float], ret_4h: list[float]) -> float:
    """Reversal speed: higher if token fades quickly (1h/4h).

    Looks at how negative the early returns are relative to the pump.
    Score 0-100 where 100 = very fast reverter.
    """
    if not ret_1h:
        return 50.0

    # Average magnitude of 1h negative returns
    neg_1h = [abs(x) for x in ret_1h if x < 0]
    if not neg_1h:
        return 20.0  # rarely reverses quickly

    avg_neg_1h = sum(neg_1h) / len(neg_1h)
    freq_neg_1h = len(neg_1h) / len(ret_1h)

    # Combine frequency and magnitude
    # Scale: 2% avg negative with 60% frequency -> ~70 score
    magnitude_score = min(avg_neg_1h / 3.0, 1.0)  # cap at 3% avg
    raw = freq_neg_1h * 0.6 + magnitude_score * 0.4

    return max(0.0, min(raw * 100.0, 100.0))


def _compute_cluster_sensitivity(clustered: list, unclustered: list) -> float:
    """Cluster sensitivity: compare 24h returns when clustered vs not.

    Positive = clustering improves predictive power (more negative returns).
    Returns signed value (not 0-100).
    """
    if not clustered or not unclustered:
        return 0.0

    cl_ret = [float(r[3]) for r in clustered if r[3] is not None]
    uncl_ret = [float(r[3]) for r in unclustered if r[3] is not None]

    if not cl_ret or not uncl_ret:
        return 0.0

    avg_cl = sum(cl_ret) / len(cl_ret)
    avg_uncl = sum(uncl_ret) / len(uncl_ret)

    # For shorts: more negative when clustered = positive sensitivity
    return avg_uncl - avg_cl  # positive if cluster helps


def _compute_regime_extremes(
    regime_stats: dict, min_sample: int
) -> tuple[str | None, str | None]:
    """Find best and worst regime for a symbol.

    Best = most negative avg_24h (best for shorts).
    Worst = most positive avg_24h (worst for shorts).
    """
    valid = {k: v for k, v in regime_stats.items() if v["n"] >= min_sample}
    if not valid:
        return None, None

    best = min(valid.items(), key=lambda x: x[1]["avg_24h"])
    worst = max(valid.items(), key=lambda x: x[1]["avg_24h"])

    return best[0], worst[0]


def _compute_confidence(n: int, ret_24h: list[float]) -> float:
    """Confidence score 0-100 based on sample size and consistency.

    Higher with:
    - More samples (log-scaled, diminishing returns)
    - Lower variance in returns (more predictable)
    """
    # Sample size component (log-scaled): 5->30, 10->50, 20->65, 50->80, 100->90
    size_score = min(math.log(n + 1) / math.log(101) * 100, 95.0)

    # Consistency component: inverse of coefficient of variation
    if len(ret_24h) >= 3:
        mean = sum(ret_24h) / len(ret_24h)
        variance = sum((x - mean) ** 2 for x in ret_24h) / len(ret_24h)
        std = math.sqrt(variance) if variance > 0 else 0
        # CV = std / |mean|, lower = more consistent
        if abs(mean) > 0.1:
            cv = std / abs(mean)
            consistency = max(0.0, 100.0 - cv * 20.0)
        else:
            # Mean near zero: moderate consistency (can't tell direction)
            consistency = 40.0
    else:
        consistency = 30.0

    # Weight: 60% sample size, 40% consistency
    return max(0.0, min(size_score * 0.6 + consistency * 0.4, 100.0))


def _classify_personality(
    trend_follow: float,
    mean_reversion: float,
    reversal_speed: float,
    cluster_sensitivity: float,
    confidence: float,
) -> str:
    """Deterministic personality label from scores.

    Returns one of:
    - "fade-prone": strong mean reversion tendency
    - "trend-following": tends to continue pumping
    - "fast-reverting": fades very quickly (1h/4h)
    - "cluster-sensitive": behaves differently when clustered
    - "mixed": no clear pattern
    """
    if confidence < 25:
        return "mixed"

    # Strong mean reversion: mean_reversion > 65 and trend_follow < 45
    if mean_reversion >= 65 and trend_follow < 45:
        if reversal_speed >= 65:
            return "fast-reverting"
        return "fade-prone"

    # Strong trend following: trend_follow > 60 and mean_reversion < 40
    if trend_follow >= 60 and mean_reversion < 40:
        return "trend-following"

    # Cluster-sensitive: large absolute cluster sensitivity
    if abs(cluster_sensitivity) > 3.0 and confidence >= 40:
        return "cluster-sensitive"

    return "mixed"


def _round_or_none(val: float | None, digits: int) -> float | None:
    if val is None:
        return None
    return round(val, digits)


def _upsert_personality(conn, tp: TokenPersonality) -> None:
    """Insert or update a token personality row."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO token_personality_memory (
                       symbol, sample_size, sample_size_clustered, sample_size_unclustered,
                       win_1h, win_4h, win_24h,
                       avg_ret_1h, avg_ret_4h, avg_ret_24h,
                       avg_mfe_24h, avg_mae_24h,
                       trend_follow_score, mean_reversion_score,
                       reversal_speed_score, cluster_sensitivity,
                       best_regime, worst_regime,
                       confidence_score, personality_label, updated_at
                   ) VALUES (
                       %s, %s, %s, %s,
                       %s, %s, %s,
                       %s, %s, %s,
                       %s, %s,
                       %s, %s, %s, %s,
                       %s, %s,
                       %s, %s, now()
                   )
                   ON CONFLICT (symbol) DO UPDATE SET
                       sample_size = EXCLUDED.sample_size,
                       sample_size_clustered = EXCLUDED.sample_size_clustered,
                       sample_size_unclustered = EXCLUDED.sample_size_unclustered,
                       win_1h = EXCLUDED.win_1h,
                       win_4h = EXCLUDED.win_4h,
                       win_24h = EXCLUDED.win_24h,
                       avg_ret_1h = EXCLUDED.avg_ret_1h,
                       avg_ret_4h = EXCLUDED.avg_ret_4h,
                       avg_ret_24h = EXCLUDED.avg_ret_24h,
                       avg_mfe_24h = EXCLUDED.avg_mfe_24h,
                       avg_mae_24h = EXCLUDED.avg_mae_24h,
                       trend_follow_score = EXCLUDED.trend_follow_score,
                       mean_reversion_score = EXCLUDED.mean_reversion_score,
                       reversal_speed_score = EXCLUDED.reversal_speed_score,
                       cluster_sensitivity = EXCLUDED.cluster_sensitivity,
                       best_regime = EXCLUDED.best_regime,
                       worst_regime = EXCLUDED.worst_regime,
                       confidence_score = EXCLUDED.confidence_score,
                       personality_label = EXCLUDED.personality_label,
                       updated_at = now()""",
                (
                    tp.symbol, tp.sample_size, tp.sample_size_clustered,
                    tp.sample_size_unclustered,
                    tp.win_1h, tp.win_4h, tp.win_24h,
                    tp.avg_ret_1h, tp.avg_ret_4h, tp.avg_ret_24h,
                    tp.avg_mfe_24h, tp.avg_mae_24h,
                    tp.trend_follow_score, tp.mean_reversion_score,
                    tp.reversal_speed_score, tp.cluster_sensitivity,
                    tp.best_regime, tp.worst_regime,
                    tp.confidence_score, tp.personality_label,
                ),
            )
        conn.commit()
    except Exception as exc:
        log.warning("_upsert_personality failed for %s: %s", tp.symbol, exc)
        try:
            conn.rollback()
        except Exception:
            pass


# ── Conviction Adjustment ─────────────────────────────────────


def compute_conviction_adjustment(
    pg_dsn: str,
    symbol: str,
    *,
    max_boost: float = 6.0,
    max_penalty: float = 6.0,
    regime_code: str = "",
    confidence_min: float = 40.0,
) -> tuple[float, str]:
    """Compute conviction adjustment from token personality.

    This is the FINAL refinement layer in conviction scoring, applied after
    all other components:
      1. Base composite score (from adaptive-weighted signals)
      2. Historical edge (alert_outcomes win rates)
      3. BTC regime context (EMA trend)
      4. Liquidity strength
      5. Geo alignment
      6. Token memory (THIS) — bounded ±max_boost/max_penalty additive

    Token memory must NEVER dominate conviction. It is a small refinement
    that leverages learned per-token behavioral patterns.

    Returns (adjustment, reason).
    Adjustment is positive for boost, negative for penalty.
    Bounded by max_boost/max_penalty.
    """
    tp = get_personality(pg_dsn, symbol)
    if tp is None:
        return 0.0, ""

    # Skip low-confidence personalities
    if tp.confidence_score < confidence_min:
        return 0.0, ""

    # Scale by confidence (0-1)
    conf_scale = tp.confidence_score / 100.0

    adjustment = 0.0
    reason = ""

    # Label-based adjustment
    if tp.personality_label in ("fade-prone", "fast-reverting"):
        # Boost: token tends to reverse -> good for shorts
        raw = tp.mean_reversion_score / 100.0 * max_boost
        adjustment = raw * conf_scale
        reason = f"token fades (mr={tp.mean_reversion_score:.0f})"
    elif tp.personality_label == "trend-following":
        # Penalty: token tends to continue pumping -> bad for shorts
        raw = tp.trend_follow_score / 100.0 * max_penalty
        adjustment = -raw * conf_scale
        reason = f"token trends (tf={tp.trend_follow_score:.0f})"
    elif tp.personality_label == "cluster-sensitive":
        # Small boost if cluster sensitivity is positive
        if tp.cluster_sensitivity > 0:
            raw = min(tp.cluster_sensitivity / 5.0, 1.0) * max_boost * 0.5
            adjustment = raw * conf_scale
            reason = f"cluster-sensitive (+{tp.cluster_sensitivity:.1f})"
    else:
        # mixed: label-based adjustment is zero, but regime match may still apply
        pass

    # Regime-matching adjustment (additive, modest)
    regime_adj = 0.0
    if regime_code and tp.best_regime and tp.worst_regime:
        if regime_code == tp.best_regime and regime_code != tp.worst_regime:
            # Current regime is this token's best -> modest boost
            regime_adj = 1.5 * conf_scale
            reason = f"{reason}, best regime" if reason else f"best regime ({regime_code})"
        elif regime_code == tp.worst_regime and regime_code != tp.best_regime:
            # Current regime is this token's worst -> modest penalty
            regime_adj = -1.5 * conf_scale
            reason = f"{reason}, worst regime" if reason else f"worst regime ({regime_code})"

    adjustment += regime_adj

    # Clamp total adjustment
    adjustment = max(-max_penalty, min(adjustment, max_boost))

    if adjustment == 0.0:
        return 0.0, ""

    return round(adjustment, 2), reason


# ── Alert message helper ─────────────────────────────────────


def format_personality_line(pg_dsn: str, symbol: str) -> str | None:
    """Return a compact one-line personality summary for alert messages.

    Returns None if no personality data available.
    """
    tp = get_personality(pg_dsn, symbol)
    if tp is None:
        return None

    label_map = {
        "fade-prone": "tends to fade",
        "fast-reverting": "fast reverter",
        "trend-following": "tends to follow through",
        "cluster-sensitive": "cluster-sensitive",
        "mixed": "mixed behavior",
    }

    text = label_map.get(tp.personality_label, tp.personality_label)
    return f"Token memory: {text} (conf {tp.confidence_score:.0f})"
