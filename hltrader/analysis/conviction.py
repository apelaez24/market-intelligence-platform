"""Phase 4: Conviction Score Fusion Layer.

Deterministic meta-layer that combines:
  1. Base composite score (Phase 2, using adaptive weights from Phase 6)
  2. Historical edge (Phase 3 alert_outcomes)
  3. BTC regime context (trend filter)
  4. Liquidity strength
  5. Geo severity alignment
  6. Token personality memory (Phase 7) — bounded final refinement

Component ordering (Part 5 contract):
  Adaptive weights (Phase 6) are applied BEFORE scoring in the watcher.
  Components 1-5 are combined via weighted sum.
  Token memory (6) is applied AFTER the weighted sum as a bounded
  additive adjustment (max ±6 points). It must never dominate conviction.

Returns conviction_score 0-100. No ML, no paid APIs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import psycopg2

log = logging.getLogger(__name__)

# ── TTL Cache ────────────────────────────────────────────────

_cache: dict[str, tuple[float, object]] = {}  # key -> (expires_at, value)
_DEFAULT_CACHE_TTL = 600  # 10 minutes


def _cache_get(key: str) -> object | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: object, ttl: int = _DEFAULT_CACHE_TTL) -> None:
    _cache[key] = (time.time() + ttl, value)


def cache_clear() -> None:
    """Clear conviction cache (for testing)."""
    _cache.clear()


# ── 1. Base Component ────────────────────────────────────────

def _compute_base(composite_score: float) -> float:
    """Normalize composite score to 0-100.

    Composite is already 0-100 from scorer.py.
    Clamp to valid range.
    """
    return max(0.0, min(composite_score, 100.0))


# ── 2. Historical Edge ───────────────────────────────────────

def _fetch_win_rates(pg_dsn: str, cache_ttl: int = 600) -> dict:
    """Fetch 1h win rates by alert_type and symbol from alert_outcomes.

    Returns: {
        "by_type": {"NORMAL": {"win_rate": 0.62, "n": 42}, ...},
        "by_symbol": {"BTC": {"win_rate": 0.70, "n": 8}, ...},
    }

    Cached for cache_ttl seconds. Never raises.
    """
    cache_key = "conviction_win_rates"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result = {"by_type": {}, "by_symbol": {}}
    try:
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                # By alert_type
                cur.execute(
                    """SELECT alert_type,
                              COUNT(*) FILTER (WHERE eval_1h_return < 0) AS wins,
                              COUNT(*) AS total
                       FROM alert_outcomes
                       WHERE evaluated_1h
                         AND alert_timestamp >= now() - interval '30 days'
                       GROUP BY alert_type"""
                )
                for atype, wins, total in cur.fetchall():
                    result["by_type"][atype] = {
                        "win_rate": wins / total if total > 0 else 0.5,
                        "n": total,
                    }

                # By symbol
                cur.execute(
                    """SELECT symbol,
                              COUNT(*) FILTER (WHERE eval_1h_return < 0) AS wins,
                              COUNT(*) AS total
                       FROM alert_outcomes
                       WHERE evaluated_1h
                         AND alert_timestamp >= now() - interval '30 days'
                       GROUP BY symbol"""
                )
                for sym, wins, total in cur.fetchall():
                    result["by_symbol"][sym] = {
                        "win_rate": wins / total if total > 0 else 0.5,
                        "n": total,
                    }
        finally:
            conn.close()
    except Exception as exc:
        log.warning("_fetch_win_rates failed (non-fatal): %s", exc)

    _cache_set(cache_key, result, cache_ttl)
    return result


def _compute_history(
    symbol: str,
    alert_type: str,
    pg_dsn: str,
    min_sample: int = 10,
    cache_ttl: int = 600,
    history_prior: float = 50.0,
) -> float:
    """Compute historical edge component (0-100).

    Uses symbol-specific win rate if sample >= min_sample,
    otherwise falls back to lane-level (alert_type) stats.
    If no data at all, returns history_prior (default 50 = neutral).

    Win rate mapping:
      >= 70% -> 90
      >= 60% -> 70
      >= 50% -> 50  (neutral)
      >= 40% -> 30
      >= 30% -> 15
      < 30%  -> 5   (heavy penalty)
    """
    rates = _fetch_win_rates(pg_dsn, cache_ttl)

    # Try symbol-specific first
    sym_data = rates["by_symbol"].get(symbol)
    if sym_data and sym_data["n"] >= min_sample:
        wr = sym_data["win_rate"]
    else:
        # Fall back to lane-level
        type_data = rates["by_type"].get(alert_type)
        if type_data and type_data["n"] > 0:
            wr = type_data["win_rate"]
        else:
            # No data at all — return prior (neutral by default)
            return history_prior

    return _win_rate_to_score(wr)


def _win_rate_to_score(wr: float) -> float:
    """Map win rate (0.0-1.0) to component score (0-100)."""
    if wr >= 0.70:
        return 90.0
    elif wr >= 0.60:
        return 70.0
    elif wr >= 0.50:
        return 50.0
    elif wr >= 0.40:
        return 30.0
    elif wr >= 0.30:
        return 15.0
    else:
        return 5.0


# ── 3. BTC Regime Context ────────────────────────────────────

def _fetch_btc_regime(pg_dsn_coinbase: str, cache_ttl: int = 600) -> str:
    """Classify BTC regime from 4h candles (using 1h candles, aggregated).

    Returns: "strong_uptrend", "strong_downtrend", or "chop".
    Cached for cache_ttl seconds. Never raises.
    """
    cache_key = "conviction_btc_regime"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    regime = "chop"  # default
    try:
        conn = psycopg2.connect(pg_dsn_coinbase)
        try:
            with conn.cursor() as cur:
                # Fetch last 100 hourly candles (~4 days, enough for 20-period EMA on 4h)
                cur.execute(
                    """SELECT close FROM btcusd_1h
                       ORDER BY datetime DESC LIMIT 100"""
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 24:
            _cache_set(cache_key, regime, cache_ttl)
            return regime

        # Reverse to chronological order (cast Decimal->float for numeric columns)
        closes = [float(r[0]) for r in reversed(rows)]

        # Aggregate to 4h candles (average of 4 consecutive closes)
        candles_4h = []
        for i in range(0, len(closes) - 3, 4):
            candles_4h.append(sum(closes[i:i+4]) / 4.0)

        if len(candles_4h) < 20:
            _cache_set(cache_key, regime, cache_ttl)
            return regime

        # 20-period EMA on 4h candles
        ema = _compute_ema(candles_4h, 20)

        if len(ema) < 5:
            _cache_set(cache_key, regime, cache_ttl)
            return regime

        # Slope over last 5 EMA points
        recent_ema = ema[-5:]
        slope = (recent_ema[-1] - recent_ema[0]) / recent_ema[0] * 100.0

        # Classification
        if slope > 1.0:
            regime = "strong_uptrend"
        elif slope < -1.0:
            regime = "strong_downtrend"
        else:
            regime = "chop"

    except Exception as exc:
        log.warning("_fetch_btc_regime failed (non-fatal): %s", exc)

    _cache_set(cache_key, regime, cache_ttl)
    return regime


def _compute_ema(values: list[float], period: int) -> list[float]:
    """Compute EMA series. Returns list same length as input minus (period-1)."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(values[:period]) / period]  # SMA seed
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def _compute_regime_score(regime: str) -> float:
    """Map BTC regime to short-alert score component (0-100).

    Short-biased alerts: downtrend = boost, uptrend = penalty.
    """
    if regime == "strong_downtrend":
        return 80.0  # favorable for shorts
    elif regime == "strong_uptrend":
        return 20.0  # unfavorable for shorts
    else:
        return 50.0  # neutral (chop)


# ── 4. Liquidity Component ───────────────────────────────────

def _compute_liquidity_component(liquidity_score: float) -> float:
    """Normalize liquidity score to 0-100 conviction component.

    liquidity_score is already 0-1 from scorer.py.
    High liquidity -> boost, microcap -> penalty.
    """
    # Scale 0-1 to 0-100
    return max(0.0, min(liquidity_score * 100.0, 100.0))


# ── 5. Geo Alignment ─────────────────────────────────────────

# Categories that map to crypto/risk-off relevance
_GEO_RELEVANT_WATCHLIST = {"energy", "risk_off", "defense", "crypto_sentiment", "bonds"}


def _fetch_active_geo_severity(pg_dsn: str, cache_ttl: int = 600) -> list[dict]:
    """Fetch geo events with severity > 80 from last 4 hours.

    Returns list of {"severity": int, "watchlist": list[str]}.
    Cached. Never raises.
    """
    cache_key = "conviction_geo_events"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    events = []
    try:
        conn = psycopg2.connect(pg_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT severity, market_watchlist
                       FROM geo_events
                       WHERE severity > 80
                         AND created_at >= now() - interval '4 hours'
                       ORDER BY severity DESC
                       LIMIT 10"""
                )
                for sev, watchlist in cur.fetchall():
                    wl = watchlist if isinstance(watchlist, list) else []
                    events.append({"severity": sev, "watchlist": wl})
        finally:
            conn.close()
    except Exception as exc:
        log.warning("_fetch_active_geo_severity failed (non-fatal): %s", exc)

    _cache_set(cache_key, events, cache_ttl)
    return events


def _compute_geo_component(pg_dsn: str, cache_ttl: int = 600) -> float:
    """Compute geo alignment score (0-100).

    If high-severity geo events exist with relevant watchlist categories,
    boost conviction (shorts benefit from risk-off).
    """
    events = _fetch_active_geo_severity(pg_dsn, cache_ttl)
    if not events:
        return 50.0  # neutral — no active geo risk

    # Check if any event has relevant watchlist alignment
    max_alignment = 0.0
    for ev in events:
        wl_set = set(ev["watchlist"])
        overlap = wl_set & _GEO_RELEVANT_WATCHLIST
        if overlap:
            # More overlap categories = stronger alignment
            alignment = min(len(overlap) / 3.0, 1.0)  # max at 3+ categories
            severity_factor = ev["severity"] / 100.0
            score = alignment * severity_factor
            max_alignment = max(max_alignment, score)

    if max_alignment > 0:
        # Scale: 0.0 -> 50 (neutral), 1.0 -> 90 (strong boost)
        return 50.0 + max_alignment * 40.0
    return 50.0  # neutral


# ── 6. Token Personality Memory ───────────────────────────────
#
# This is the FINAL refinement layer, applied AFTER the weighted sum of
# components 1-5. It leverages learned per-token behavioral patterns
# from Phase 7's token_personality_memory table.
#
# Ordering contract (Part 5):
#   1. Base composite score (from adaptive-weighted signals, Phase 6)
#   2. Historical edge (alert_outcomes win rates, Phase 3)
#   3. BTC regime context (coinbase EMA trend)
#   4. Liquidity strength
#   5. Geo alignment
#   6. Token memory (THIS) — bounded ±max_boost/max_penalty additive
#
# Token memory must NEVER dominate the overall conviction score.
# Max absolute adjustment is capped at ±6 points (configurable).

def _compute_token_memory_refinement(
    pg_dsn: str,
    symbol: str,
    regime_code: str = "",
    *,
    max_boost: float = 6.0,
    max_penalty: float = 6.0,
    confidence_min: float = 40.0,
) -> tuple[float, str]:
    """Compute bounded token memory refinement for conviction.

    Delegates to token_memory.compute_conviction_adjustment() which
    handles personality lookup, label-based scoring, regime matching,
    and confidence scaling.

    Returns (adjustment, reason). Never raises.
    """
    try:
        from hltrader.analysis.token_memory import compute_conviction_adjustment
        return compute_conviction_adjustment(
            pg_dsn, symbol,
            max_boost=max_boost,
            max_penalty=max_penalty,
            regime_code=regime_code,
            confidence_min=confidence_min,
        )
    except Exception as exc:
        log.warning("token_memory refinement failed for %s (non-fatal): %s", symbol, exc)
        return 0.0, ""


# ── Reason Generation ─────────────────────────────────────────

_COMPONENT_LABELS = {
    "base": "strong composite",
    "history": "proven win rate",
    "regime": "BTC regime aligned",
    "liquidity": "high liquidity",
    "geo": "geo risk-off boost",
}

_COMPONENT_LABELS_NEGATIVE = {
    "base": "weak composite",
    "history": "poor/no history",
    "regime": "BTC uptrend headwind",
    "liquidity": "low liquidity",
    "geo": "no geo catalyst",
}


def _generate_reasons(
    components: dict[str, float],
    conviction: float,
    threshold: float,
    token_memory_reason: str = "",
) -> list[str]:
    """Generate top-2 reasons for allow/block decision.

    For allowed alerts: pick the 2 strongest components (highest deviation above 50).
    For blocked alerts: pick the 2 weakest components (most below 50).
    If token memory contributed, it can replace the weakest reason.
    """
    allowed = conviction >= threshold

    # Compute deviation from neutral (50) for each component (exclude token_memory)
    deviations = []
    for name, score in components.items():
        if name == "token_memory":
            continue
        deviation = score - 50.0
        deviations.append((name, score, deviation))

    if allowed:
        # Sort by deviation descending (strongest contributors)
        deviations.sort(key=lambda x: x[2], reverse=True)
        reasons = []
        for name, score, dev in deviations[:2]:
            if dev > 0:
                reasons.append(_COMPONENT_LABELS.get(name, name))
            else:
                reasons.append(_COMPONENT_LABELS_NEGATIVE.get(name, name))
    else:
        # Sort by deviation ascending (weakest / most negative)
        deviations.sort(key=lambda x: x[2])
        reasons = []
        for name, score, dev in deviations[:2]:
            if dev < 0:
                reasons.append(_COMPONENT_LABELS_NEGATIVE.get(name, name))
            else:
                reasons.append(_COMPONENT_LABELS.get(name, name))

    # If token memory contributed meaningfully, add as additional reason
    if token_memory_reason:
        reasons.append(token_memory_reason)

    return reasons[:3]


# ── Main Entry Point ─────────────────────────────────────────

def compute_conviction(
    symbol: str,
    alert_type: str,
    composite_score: float,
    pump_score: float,
    funding_score: float,
    oi_score: float,
    accel_score: float,
    liquidity: float,
    alert_timestamp: datetime,
    *,
    pg_dsn: str = "",
    pg_dsn_coinbase: str = "",
    w_base: float = 40.0,
    w_history: float = 20.0,
    w_regime: float = 20.0,
    w_liquidity: float = 10.0,
    w_geo: float = 10.0,
    min_sample: int = 10,
    cache_ttl: int = 600,
    history_prior: float = 50.0,
    conviction_min: float = 55.0,
    # Phase 7: Token memory integration
    regime_code: str = "",
    token_memory_enabled: bool = False,
    token_memory_max_boost: float = 6.0,
    token_memory_max_penalty: float = 6.0,
    token_memory_confidence_min: float = 40.0,
) -> dict:
    """Compute conviction score with component breakdown.

    Components 1-5 are combined via weighted sum. Token memory (6) is
    applied after the weighted sum as a bounded additive refinement.

    Returns dict with:
      - conviction: float (0-100)
      - tier: str ("A", "B", "C", or "")
      - components: dict of individual scores
      - regime: str
      - reasons: list[str] (top 2-3 reasons for allow/block)
      - allowed: bool (whether conviction >= threshold)
      - token_memory_adj: float (token memory adjustment applied)
    """
    # Normalize weights
    w_total = w_base + w_history + w_regime + w_liquidity + w_geo
    if w_total <= 0:
        w_total = 100.0

    # Compute components 1-5
    base = _compute_base(composite_score)

    history = _compute_history(
        symbol, alert_type, pg_dsn, min_sample=min_sample, cache_ttl=cache_ttl,
        history_prior=history_prior,
    ) if pg_dsn else history_prior

    regime = _fetch_btc_regime(pg_dsn_coinbase, cache_ttl) if pg_dsn_coinbase else "chop"
    regime_score = _compute_regime_score(regime)

    liq = _compute_liquidity_component(liquidity)

    geo = _compute_geo_component(pg_dsn, cache_ttl) if pg_dsn else 50.0

    # Weighted sum of components 1-5
    conviction = (
        base * (w_base / w_total)
        + history * (w_history / w_total)
        + regime_score * (w_regime / w_total)
        + liq * (w_liquidity / w_total)
        + geo * (w_geo / w_total)
    )

    conviction = max(0.0, min(conviction, 100.0))

    # Component 6: Token memory — bounded additive refinement (AFTER weighted sum)
    tm_adj = 0.0
    tm_reason = ""
    if token_memory_enabled and pg_dsn:
        tm_adj, tm_reason = _compute_token_memory_refinement(
            pg_dsn, symbol,
            regime_code=regime_code,
            max_boost=token_memory_max_boost,
            max_penalty=token_memory_max_penalty,
            confidence_min=token_memory_confidence_min,
        )
        if tm_adj != 0.0:
            conviction = max(0.0, min(conviction + tm_adj, 100.0))
            log.info("token_memory conviction adj %s: %+.1f (%s)", symbol, tm_adj, tm_reason)

    tier = conviction_tier(conviction)

    components = {
        "base": round(base, 1),
        "history": round(history, 1),
        "regime": round(regime_score, 1),
        "liquidity": round(liq, 1),
        "geo": round(geo, 1),
        "token_memory": round(tm_adj, 1),
    }

    reasons = _generate_reasons(components, conviction, conviction_min, tm_reason)
    allowed = conviction >= conviction_min

    return {
        "conviction": round(conviction, 1),
        "tier": tier,
        "regime": regime,
        "components": components,
        "reasons": reasons,
        "allowed": allowed,
        "token_memory_adj": round(tm_adj, 1),
    }


def conviction_tier(score: float) -> str:
    """Map conviction score to tier label."""
    if score >= 75:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 55:
        return "C"
    return ""
