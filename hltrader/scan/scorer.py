"""Score coins as short candidates based on pump, funding, and OI signals."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Asset classification
_COMMODITIES = {"PAXG"}
_STOCKS: set[str] = set()  # None on perps yet, reserved for future


def classify_asset(coin: str) -> str:
    """Return 'Commodity', 'Stock', or 'Crypto' for a given coin name."""
    if coin in _COMMODITIES:
        return "Commodity"
    if coin in _STOCKS:
        return "Stock"
    return "Crypto"


@dataclass
class ShortCandidate:
    coin: str
    asset_type: str
    price: float
    pct_24h: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    pump_score: float
    funding_score: float
    oi_score: float
    composite: float
    # Phase 2 fields (optional, backward-compatible defaults)
    ret_1h: float | None = None
    ret_4h: float | None = None
    accel_score: float = 0.0
    liquidity_score: float = 0.0
    tier: str = ""  # "A", "B", "C", or "" (unset)
    conviction_score: float | None = None  # Phase 4: conviction meta-score
    conviction_reasons: list | None = None  # Phase 4.1: top 2 reasons
    conviction_allowed: bool | None = None  # Phase 4.1: whether allowed by threshold


def compute_pump_score(pct_24h: float, cap: float = 40.0, exp: float = 1.0) -> float:
    """Compute pump score with configurable cap and exponent.

    Linear when exp=1.0 (legacy). Convex curve when exp>1 rewards
    larger pumps more while compressing small ones.
    """
    p = max(pct_24h, 0.0)
    pump_norm = min(p / cap, 1.0)
    return pump_norm ** exp


def compute_liquidity_cutoff(
    values: list[float],
    keep_pct: float = 0.70,
) -> float:
    """Compute the percentile cutoff value for a liquidity filter.

    keep_pct=0.70 means keep the top 70%, so the cutoff is at
    the 30th percentile (P30) of the distribution.

    Returns 0.0 if fewer than 2 values or keep_pct >= 1.0.
    """
    if keep_pct >= 1.0 or len(values) < 2:
        return 0.0
    positive = sorted(v for v in values if v > 0)
    if not positive:
        return 0.0
    drop_pct = 1.0 - keep_pct
    idx = int(math.floor(len(positive) * drop_pct))
    idx = min(idx, len(positive) - 1)
    return positive[idx]


def compute_accel_score(
    ret_1h: float | None,
    ret_4h: float | None,
    a0: float = 0.0,
    a1: float = 2.0,
) -> float:
    """Compute momentum acceleration proxy.

    accel = ret_1h - (ret_4h / 4)
    Positive means the pump is speeding up vs its average hourly pace.
    Returns 0-1 clamped score.

    If either return is None, returns 0.0 (no data).
    """
    if ret_1h is None or ret_4h is None:
        return 0.0
    accel = ret_1h - (ret_4h / 4.0)
    return max(0.0, min((accel - a0) / a1, 1.0)) if a1 > 0 else 0.0


def compute_liquidity_score(volume_24h: float, open_interest: float) -> float:
    """Compute a 0-1 liquidity score using log-scaled volume + OI.

    Used as tie-breaker in ranking so junk doesn't rise on pump alone.
    """
    vol_log = math.log10(max(volume_24h, 1.0))   # ~6 for $1M, ~9 for $1B
    oi_log = math.log10(max(open_interest, 1.0))  # ~6 for $1M
    # Normalize: $1M vol = 0.0, $1B vol = 1.0  (log range 6-9)
    vol_norm = max(0.0, min((vol_log - 6.0) / 3.0, 1.0))
    oi_norm = max(0.0, min((oi_log - 5.0) / 4.0, 1.0))   # 100K-10B range
    return (vol_norm * 0.6 + oi_norm * 0.4)


def compute_percentile_cutoff(values: list[float], keep: float) -> float:
    """Compute the percentile cutoff: keep top `keep` fraction.

    keep=0.30 means keep top 30%, so cutoff is at the 70th percentile.
    Returns 0.0 if < 10 values (skip small-sample filtering) or keep >= 1.0.
    """
    if keep >= 1.0 or len(values) < 10:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(math.floor(len(sorted_vals) * (1.0 - keep)))
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def assign_tier(composite: float) -> str:
    """Assign alert tier based on composite score."""
    if composite >= 35:
        return "A"
    elif composite >= 25:
        return "B"
    elif composite >= 20:
        return "C"
    return ""


def score_short_candidates(
    snapshots: list[dict],
    *,
    min_score: float = 10.0,
    min_volume: float = 1_000_000.0,
    min_pct_24h: float = 5.0,
    min_funding: float = -1.0,
    min_oi: float = 0.0,
    pump_cap: float = 40.0,
    pump_exp: float = 1.0,
    liquidity_percentile_keep: float = 1.0,
    liquidity_metric: str = "auto",
    # Phase 2 params
    w_pump: float = 50.0,
    w_funding: float = 30.0,
    w_oi: float = 20.0,
    w_accel: float = 0.0,
    accel_a0: float = 0.0,
    accel_a1: float = 2.0,
    pump_percentile_keep: float = 1.0,
    score_percentile_keep: float = 1.0,
    percentile_mode: str = "pump",
    returns_cache: dict | None = None,
) -> list[ShortCandidate]:
    """Score and filter short candidates from market snapshots.

    Only coins with positive 24h change qualify. Positive funding boosts
    the score but is not required. OI is normalized against the 90th
    percentile across all coins.
    """
    # Compute OI 90th percentile for normalization
    all_oi = sorted(s["open_interest"] for s in snapshots if s["open_interest"] > 0)
    if all_oi:
        idx = int(len(all_oi) * 0.9)
        idx = min(idx, len(all_oi) - 1)
        oi_p90 = all_oi[idx]
    else:
        oi_p90 = 1.0  # avoid division by zero

    # Phase 1: Pre-filter by hard floors
    pre_filtered: list[dict] = []
    for s in snapshots:
        pct = s["pct_24h"]
        vol = s["volume_24h"]
        oi = s["open_interest"]

        if pct <= 0:
            continue
        if pct < min_pct_24h:
            continue
        if vol < min_volume:
            continue
        if s["funding_rate"] < min_funding:
            continue
        if oi < min_oi:
            continue
        pre_filtered.append(s)

    # Phase 2: Liquidity percentile filter
    if liquidity_percentile_keep < 1.0 and len(pre_filtered) >= 2:
        if liquidity_metric == "oi":
            liq_values = [s["open_interest"] for s in pre_filtered]
        elif liquidity_metric == "volume":
            liq_values = [s["volume_24h"] for s in pre_filtered]
        else:  # "auto": prefer OI if most have it, else volume
            oi_count = sum(1 for s in pre_filtered if s["open_interest"] > 0)
            if oi_count >= len(pre_filtered) * 0.5:
                liq_values = [s["open_interest"] for s in pre_filtered]
            else:
                liq_values = [s["volume_24h"] for s in pre_filtered]

        cutoff = compute_liquidity_cutoff(liq_values, liquidity_percentile_keep)
        if cutoff > 0:
            if liquidity_metric == "volume":
                pre_filtered = [s for s in pre_filtered if s["volume_24h"] >= cutoff]
            elif liquidity_metric == "oi":
                pre_filtered = [s for s in pre_filtered if s["open_interest"] >= cutoff]
            else:
                # Auto: use whichever metric was selected
                if oi_count >= len(pre_filtered) * 0.5:
                    pre_filtered = [s for s in pre_filtered if s["open_interest"] >= cutoff]
                else:
                    pre_filtered = [s for s in pre_filtered if s["volume_24h"] >= cutoff]

    # Phase 3: Score with configurable weights + acceleration
    rc = returns_cache or {}
    candidates: list[ShortCandidate] = []
    for s in pre_filtered:
        pct = s["pct_24h"]
        funding = s["funding_rate"]
        oi = s["open_interest"]
        coin = s["coin"]

        pump_sc = compute_pump_score(pct, cap=pump_cap, exp=pump_exp)
        funding_sc = min(funding / 0.01, 1.0) if funding > 0 else 0.0
        oi_sc = min(oi / oi_p90, 1.0) if oi_p90 > 0 else 0.0

        # Acceleration: use cached 1h/4h returns if available
        ret_1h = rc.get(coin, {}).get("ret_1h")
        ret_4h = rc.get(coin, {}).get("ret_4h")
        accel_sc = compute_accel_score(ret_1h, ret_4h, a0=accel_a0, a1=accel_a1)

        liq_sc = compute_liquidity_score(s["volume_24h"], oi)

        composite = (pump_sc * w_pump + funding_sc * w_funding
                     + oi_sc * w_oi + accel_sc * w_accel)

        if composite < min_score:
            continue

        candidates.append(ShortCandidate(
            coin=coin,
            asset_type=classify_asset(coin),
            price=s["price"],
            pct_24h=pct,
            funding_rate=funding,
            open_interest=oi,
            volume_24h=s["volume_24h"],
            pump_score=pump_sc,
            funding_score=funding_sc,
            oi_score=oi_sc,
            composite=composite,
            ret_1h=ret_1h,
            ret_4h=ret_4h,
            accel_score=accel_sc,
            liquidity_score=liq_sc,
            tier=assign_tier(composite),
        ))

    # Sort by composite desc, then by liquidity_score desc as tie-breaker
    candidates.sort(key=lambda c: (c.composite, c.liquidity_score), reverse=True)

    # Phase 4: Dynamic percentile filter (skip if < 10 candidates)
    if len(candidates) >= 10:
        if percentile_mode in ("pump", "both") and pump_percentile_keep < 1.0:
            cutoff = compute_percentile_cutoff(
                [c.pump_score for c in candidates], pump_percentile_keep)
            if cutoff > 0:
                before = len(candidates)
                candidates = [c for c in candidates if c.pump_score >= cutoff]
                log.info("percentile_pump: cutoff=%.3f, %d->%d",
                         cutoff, before, len(candidates))

        if percentile_mode in ("score", "both") and score_percentile_keep < 1.0:
            cutoff = compute_percentile_cutoff(
                [c.composite for c in candidates], score_percentile_keep)
            if cutoff > 0:
                before = len(candidates)
                candidates = [c for c in candidates if c.composite >= cutoff]
                log.info("percentile_score: cutoff=%.1f, %d->%d",
                         cutoff, before, len(candidates))

    return candidates


# Hard cap: only candidates with composite <= this value are "good short setups"
# (lower score = stronger bearish signal)
MAX_ALERT_SCORE = 45


def filter_alert_candidates(candidates: list[ShortCandidate],
                            max_score: float = MAX_ALERT_SCORE) -> list[ShortCandidate]:
    """Filter candidates to only those with composite <= max_score.

    Our composite score represents pump strength (higher = bigger pump).
    For SHORT candidates, we want coins that pumped enough to qualify but
    NOT so much that the score is extreme (likely momentum, not reversion).
    Scores <= 45 represent moderate pumps with favorable funding/OI — better
    short setups than extreme 80+ outliers.
    """
    kept: list[ShortCandidate] = []
    for c in candidates:
        if c.composite > max_score:
            log.info("Candidate filtered: %s score %.1f > %.0f", c.coin, c.composite, max_score)
        else:
            kept.append(c)
    return kept


def classify_extreme_candidates(
    candidates: list[ShortCandidate],
    *,
    score_threshold: float = 45.0,
    pct_threshold: float = 20.0,
) -> tuple[list[ShortCandidate], list[ShortCandidate]]:
    """Split candidates into (normal, extreme) lists.

    A candidate is extreme if:
      - composite > score_threshold, OR
      - pct_24h >= pct_threshold

    Normal candidates have composite <= score_threshold AND pct_24h < pct_threshold.
    """
    normal: list[ShortCandidate] = []
    extreme: list[ShortCandidate] = []
    for c in candidates:
        if c.composite > score_threshold or c.pct_24h >= pct_threshold:
            extreme.append(c)
        else:
            normal.append(c)
    return normal, extreme
