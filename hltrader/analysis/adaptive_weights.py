"""Phase 6 / 6.1: Adaptive Signal Weights.

Learns optimal signal weights from historical alert outcomes using
Pearson correlation. Runs daily, stores per-regime weights with EMA
smoothing and daily-delta clamping to prevent overfitting/whipsaw.

Pipeline ordering (deterministic):
  1. Pearson correlation per signal vs negated 24h return
  2. Normalize correlations to sum-to-1 weights
  3. EMA smooth against prior weights
  4. Clamp daily deltas vs prior weights
  5. Re-normalize to sum-to-1

No external dependencies — pure Python correlation computation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone

import psycopg2

log = logging.getLogger(__name__)

# Signal columns in alert_outcomes (order matters — matches weight output)
SIGNAL_COLUMNS = ("pump_score", "oi_score", "funding_score", "accel_score")
WEIGHT_KEYS = ("pump_weight", "oi_weight", "funding_weight", "accel_weight")

# Default weights (normalized to sum=1, matching env defaults 40/30/20/10)
DEFAULT_WEIGHTS = {
    "pump_weight": 0.40,
    "oi_weight": 0.20,
    "funding_weight": 0.30,
    "accel_weight": 0.10,
}

GLOBAL_REGIME = "GLOBAL"


@dataclass
class AdaptiveWeights:
    """Learned signal weights."""
    pump_weight: float
    oi_weight: float
    funding_weight: float
    accel_weight: float
    regime_code: str
    sample_size: int
    computed_date: date


@dataclass
class WeightsSelection:
    """Result of get_weights_for_regime — weights + metadata."""
    weights: dict[str, float]   # scaled to sum=100 (w_pump, w_oi, w_funding, w_accel)
    source: str                 # "REGIME" | "GLOBAL" | "STATIC"
    asof_date: date | None
    sample_size: int
    regime_code: str | None


# ── Pure Python Stats ────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Normalize weights so they sum to 1.0. All weights >= 0."""
    clamped = {k: max(v, 0.0) for k, v in raw.items()}
    total = sum(clamped.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in clamped.items()}


def _ema_smooth(
    old: dict[str, float],
    new: dict[str, float],
    alpha: float = 0.2,
) -> dict[str, float]:
    """EMA smooth: result = old * (1-alpha) + new * alpha."""
    return {
        k: old.get(k, DEFAULT_WEIGHTS[k]) * (1 - alpha) + new.get(k, DEFAULT_WEIGHTS[k]) * alpha
        for k in WEIGHT_KEYS
    }


def _clamp_deltas(
    old: dict[str, float],
    new: dict[str, float],
    max_delta: float = 0.08,
) -> dict[str, float]:
    """Clamp per-weight daily change to [-max_delta, +max_delta].

    Prevents single-day overreaction. Applied after EMA, before final
    re-normalization.
    """
    result = {}
    for k in WEIGHT_KEYS:
        old_v = old.get(k, DEFAULT_WEIGHTS[k])
        new_v = new.get(k, DEFAULT_WEIGHTS[k])
        delta = new_v - old_v
        clamped = max(-max_delta, min(delta, max_delta))
        result[k] = old_v + clamped
    return result


# ── Core Computation ─────────────────────────────────────────

def compute_adaptive_weights(
    pg_dsn: str,
    *,
    lookback_days: int = 7,
    ema_alpha: float = 0.2,
    min_global: int = 30,
    min_per_regime: int = 20,
    max_daily_delta: float = 0.08,
) -> list[AdaptiveWeights]:
    """Compute adaptive weights from alert_outcomes.

    Pipeline per group (GLOBAL + each regime):
      1. Fetch evaluated rows from the last N days
      2. Pearson correlation of each signal vs negated 24h return
      3. Normalize to sum-to-1
      4. EMA smooth against prior stored weights
      5. Clamp daily deltas vs prior stored weights
      6. Re-normalize to sum-to-1
      7. Store in DB

    GLOBAL uses all rows with min_global threshold.
    Per-regime uses grouped rows with min_per_regime threshold.
    """
    conn = psycopg2.connect(pg_dsn)
    try:
        rows = _fetch_outcomes(conn, lookback_days)
        today = date.today()
        results = []
        skipped_regimes = []

        # Always attempt GLOBAL first
        if len(rows) >= min_global:
            aw = _compute_for_group(conn, GLOBAL_REGIME, rows, today, ema_alpha, max_daily_delta)
            if aw:
                results.append(aw)
        else:
            log.info("adaptive_weights: GLOBAL has %d rows (need %d) — skipping", len(rows), min_global)

        # Group by regime for per-regime weights
        by_regime: dict[str, list] = {}
        for r in rows:
            rc = r["regime_code"]
            if rc and rc != GLOBAL_REGIME:
                by_regime.setdefault(rc, []).append(r)

        for regime_code, regime_rows in sorted(by_regime.items()):
            if len(regime_rows) >= min_per_regime:
                aw = _compute_for_group(conn, regime_code, regime_rows, today, ema_alpha, max_daily_delta)
                if aw:
                    results.append(aw)
            else:
                log.info("adaptive_weights: %s has %d rows (need %d) — skipping",
                         regime_code, len(regime_rows), min_per_regime)
                skipped_regimes.append((regime_code, len(regime_rows)))

        if skipped_regimes:
            log.info("adaptive_weights: skipped regimes: %s",
                     ", ".join(f"{rc}({n})" for rc, n in skipped_regimes))

        return results
    finally:
        conn.close()


def _compute_for_group(
    conn,
    regime_code: str,
    rows: list[dict],
    today: date,
    ema_alpha: float,
    max_daily_delta: float,
) -> AdaptiveWeights | None:
    """Compute weights for a single group (GLOBAL or regime). Returns None on error."""
    try:
        returns = [r["eval_24h_return"] for r in rows]
        raw_corr = {}
        for sig_col, wt_key in zip(SIGNAL_COLUMNS, WEIGHT_KEYS):
            signals = [r[sig_col] for r in rows]
            neg_returns = [-ret for ret in returns]
            raw_corr[wt_key] = _pearson(signals, neg_returns)

        # Step 1: Normalize correlations
        new_weights = _normalize_weights(raw_corr)

        # Load previous weights for EMA + clamp reference
        prev = _load_latest_weights(conn, regime_code)
        ref = prev if prev else dict(DEFAULT_WEIGHTS)

        # Step 2: EMA smooth
        smoothed = _ema_smooth(ref, new_weights, alpha=ema_alpha)

        # Step 3: Clamp daily deltas
        clamped = _clamp_deltas(ref, smoothed, max_delta=max_daily_delta)

        # Step 4: Re-normalize
        final = _normalize_weights(clamped)

        aw = AdaptiveWeights(
            pump_weight=final["pump_weight"],
            oi_weight=final["oi_weight"],
            funding_weight=final["funding_weight"],
            accel_weight=final["accel_weight"],
            regime_code=regime_code,
            sample_size=len(rows),
            computed_date=today,
        )
        _store_weights(conn, aw)

        log.info(
            "adaptive_weights: %s n=%d pump=%.3f oi=%.3f fund=%.3f accel=%.3f",
            regime_code, len(rows),
            aw.pump_weight, aw.oi_weight, aw.funding_weight, aw.accel_weight,
        )
        return aw
    except Exception as exc:
        log.warning("adaptive_weights: failed for %s: %s", regime_code, exc)
        return None


# ── DB Helpers ───────────────────────────────────────────────

def _fetch_outcomes(conn, lookback_days: int) -> list[dict]:
    """Fetch alert outcomes with nearest regime code."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT a.pump_score, a.oi_score, a.funding_score, a.accel_score,
                      a.eval_24h_return, r.regime_code
               FROM alert_outcomes a
               LEFT JOIN LATERAL (
                   SELECT regime_code
                   FROM market_regime_snapshots
                   WHERE snapshot_ts <= a.alert_timestamp
                   ORDER BY snapshot_ts DESC
                   LIMIT 1
               ) r ON true
               WHERE a.eval_24h_return IS NOT NULL
                 AND a.alert_timestamp >= now() - make_interval(days := %s)
                 AND a.pump_score IS NOT NULL
                 AND a.oi_score IS NOT NULL
                 AND a.funding_score IS NOT NULL""",
            (lookback_days,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_latest_weights(conn, regime_code: str) -> dict[str, float] | None:
    """Load most recent weights for a regime."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT pump_weight, oi_weight, funding_weight, accel_weight
               FROM adaptive_signal_weights
               WHERE regime_code = %s
               ORDER BY date DESC LIMIT 1""",
            (regime_code,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return dict(zip(WEIGHT_KEYS, row))


def _load_latest_with_meta(conn, regime_code: str) -> tuple[dict[str, float] | None, date | None, int]:
    """Load most recent weights + date + sample_size for a regime."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT pump_weight, oi_weight, funding_weight, accel_weight,
                      date, sample_size
               FROM adaptive_signal_weights
               WHERE regime_code = %s
               ORDER BY date DESC LIMIT 1""",
            (regime_code,),
        )
        row = cur.fetchone()
    if not row:
        return None, None, 0
    w = dict(zip(WEIGHT_KEYS, row[:4]))
    return w, row[4], row[5]


def _store_weights(conn, aw: AdaptiveWeights) -> None:
    """Upsert weights for (date, regime_code)."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO adaptive_signal_weights
               (date, regime_code, pump_weight, oi_weight, funding_weight, accel_weight, sample_size)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (date, regime_code)
               DO UPDATE SET pump_weight = EXCLUDED.pump_weight,
                             oi_weight = EXCLUDED.oi_weight,
                             funding_weight = EXCLUDED.funding_weight,
                             accel_weight = EXCLUDED.accel_weight,
                             sample_size = EXCLUDED.sample_size,
                             created_at = now()""",
            (aw.computed_date, aw.regime_code,
             aw.pump_weight, aw.oi_weight, aw.funding_weight, aw.accel_weight,
             aw.sample_size),
        )
    conn.commit()


# ── Weight Selection (used by watcher + state_builder) ───────

def get_weights_for_regime(
    pg_dsn: str,
    *,
    regime_code: str | None = None,
    static_weights: dict[str, float] | None = None,
) -> WeightsSelection:
    """Select best available weights for the current regime.

    Priority: regime-specific > GLOBAL > static env defaults.
    Returns WeightsSelection with scaled weights (sum=100) + metadata.
    Fail-open: returns STATIC on any DB error.
    """
    static = static_weights or {
        "w_pump": DEFAULT_WEIGHTS["pump_weight"] * 100,
        "w_oi": DEFAULT_WEIGHTS["oi_weight"] * 100,
        "w_funding": DEFAULT_WEIGHTS["funding_weight"] * 100,
        "w_accel": DEFAULT_WEIGHTS["accel_weight"] * 100,
    }
    fallback = WeightsSelection(
        weights=static, source="STATIC",
        asof_date=None, sample_size=0, regime_code=None,
    )

    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.warning("get_weights_for_regime: DB connect failed: %s", exc)
        return fallback

    try:
        # Try regime-specific first
        if regime_code and regime_code != GLOBAL_REGIME:
            w, asof, n = _load_latest_with_meta(conn, regime_code)
            if w:
                return WeightsSelection(
                    weights=_scale_to_100(w), source="REGIME",
                    asof_date=asof, sample_size=n, regime_code=regime_code,
                )

        # Fall back to GLOBAL
        w, asof, n = _load_latest_with_meta(conn, GLOBAL_REGIME)
        if w:
            return WeightsSelection(
                weights=_scale_to_100(w), source="GLOBAL",
                asof_date=asof, sample_size=n, regime_code=GLOBAL_REGIME,
            )

        return fallback
    except Exception as exc:
        log.warning("get_weights_for_regime failed (non-fatal): %s", exc)
        return fallback
    finally:
        conn.close()


def _scale_to_100(w: dict[str, float]) -> dict[str, float]:
    """Scale normalized weights (sum=1) to sum=100 for scorer compatibility."""
    return {
        "w_pump": w["pump_weight"] * 100,
        "w_oi": w["oi_weight"] * 100,
        "w_funding": w["funding_weight"] * 100,
        "w_accel": w["accel_weight"] * 100,
    }


# ── Deprecated wrapper (kept for backward compat) ────────────

def load_adaptive_weights(
    pg_dsn: str,
    *,
    regime_code: str | None = None,
) -> dict[str, float] | None:
    """Load weights (old API). Prefer get_weights_for_regime()."""
    sel = get_weights_for_regime(pg_dsn, regime_code=regime_code)
    if sel.source == "STATIC":
        return None
    return sel.weights
