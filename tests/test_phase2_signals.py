"""Tests for Phase 2: Signal-Quality Alerts (percentile, acceleration, liquidity, tiers).

Run: pytest tests/test_phase2_signals.py -v
"""

from __future__ import annotations

import pytest

from hltrader.scan.scorer import (
    ShortCandidate,
    compute_accel_score,
    compute_liquidity_score,
    compute_percentile_cutoff,
    assign_tier,
    score_short_candidates,
)


# ── compute_accel_score ───────────────────────────────────────


class TestAccelScore:
    def test_positive_acceleration(self):
        """Pump speeding up: ret_1h=3%, ret_4h=4% -> accel = 3 - 1 = 2."""
        score = compute_accel_score(ret_1h=3.0, ret_4h=4.0, a0=0.0, a1=2.0)
        # accel = 3.0 - (4.0/4) = 3.0 - 1.0 = 2.0
        # score = clamp((2.0 - 0.0) / 2.0, 0, 1) = 1.0
        assert score == pytest.approx(1.0)

    def test_negative_acceleration(self):
        """Pump slowing down: ret_1h=0.5%, ret_4h=8% -> accel = 0.5 - 2 = -1.5."""
        score = compute_accel_score(ret_1h=0.5, ret_4h=8.0, a0=0.0, a1=2.0)
        # accel = 0.5 - 2.0 = -1.5, clamped to 0
        assert score == 0.0

    def test_zero_acceleration(self):
        """Steady pace: ret_1h=1%, ret_4h=4% -> accel = 1 - 1 = 0."""
        score = compute_accel_score(ret_1h=1.0, ret_4h=4.0, a0=0.0, a1=2.0)
        assert score == 0.0

    def test_none_ret_1h(self):
        """Missing 1h data returns 0."""
        assert compute_accel_score(ret_1h=None, ret_4h=4.0) == 0.0

    def test_none_ret_4h(self):
        """Missing 4h data returns 0."""
        assert compute_accel_score(ret_1h=3.0, ret_4h=None) == 0.0

    def test_both_none(self):
        """Both missing returns 0."""
        assert compute_accel_score(ret_1h=None, ret_4h=None) == 0.0

    def test_partial_acceleration(self):
        """Moderate speed-up."""
        score = compute_accel_score(ret_1h=2.0, ret_4h=4.0, a0=0.0, a1=2.0)
        # accel = 2.0 - 1.0 = 1.0
        # score = 1.0 / 2.0 = 0.5
        assert score == pytest.approx(0.5)

    def test_custom_a0_offset(self):
        """a0 > 0 raises the bar for what counts as acceleration."""
        score = compute_accel_score(ret_1h=2.0, ret_4h=4.0, a0=0.5, a1=2.0)
        # accel = 1.0, score = (1.0 - 0.5) / 2.0 = 0.25
        assert score == pytest.approx(0.25)

    def test_clamps_to_one(self):
        """Extreme acceleration clamps to 1.0."""
        score = compute_accel_score(ret_1h=10.0, ret_4h=0.0, a0=0.0, a1=2.0)
        assert score == 1.0


# ── compute_liquidity_score ───────────────────────────────────


class TestLiquidityScore:
    def test_high_liquidity(self):
        """$1B volume + $100M OI should score near 1.0."""
        score = compute_liquidity_score(1_000_000_000, 100_000_000)
        assert score > 0.8

    def test_low_liquidity(self):
        """$100K volume + $10K OI should score near 0."""
        score = compute_liquidity_score(100_000, 10_000)
        assert score < 0.2

    def test_million_dollar_midpoint(self):
        """$1M volume is the 0-point for vol normalization."""
        score = compute_liquidity_score(1_000_000, 100_000)
        assert 0.0 <= score <= 0.5

    def test_zero_volume(self):
        """Zero volume shouldn't crash."""
        score = compute_liquidity_score(0, 0)
        assert score >= 0.0


# ── compute_percentile_cutoff ─────────────────────────────────


class TestPercentileCutoff:
    def test_keep_30_pct(self):
        """keep=0.30 on 20 values: cutoff at P70."""
        values = list(range(1, 21))  # [1..20]
        cutoff = compute_percentile_cutoff(values, keep=0.30)
        # idx = floor(20 * 0.70) = 14 -> sorted[14] = 15
        assert cutoff == 15

    def test_keep_50_pct(self):
        values = list(range(1, 21))
        cutoff = compute_percentile_cutoff(values, keep=0.50)
        # idx = floor(20 * 0.50) = 10 -> sorted[10] = 11
        assert cutoff == 11

    def test_small_sample_skips(self):
        """Fewer than 10 values: skip percentile filtering."""
        values = [1, 2, 3, 4, 5]
        assert compute_percentile_cutoff(values, keep=0.30) == 0.0

    def test_exactly_10_values(self):
        """Exactly 10 values: filter applies."""
        values = list(range(1, 11))
        cutoff = compute_percentile_cutoff(values, keep=0.30)
        assert cutoff > 0

    def test_keep_100_pct(self):
        """keep=1.0 returns 0 (no filtering)."""
        values = list(range(1, 21))
        assert compute_percentile_cutoff(values, keep=1.0) == 0.0

    def test_empty_list(self):
        assert compute_percentile_cutoff([], keep=0.30) == 0.0


# ── assign_tier ───────────────────────────────────────────────


class TestAssignTier:
    def test_tier_a(self):
        assert assign_tier(35.0) == "A"
        assert assign_tier(45.0) == "A"

    def test_tier_b(self):
        assert assign_tier(25.0) == "B"
        assert assign_tier(34.9) == "B"

    def test_tier_c(self):
        assert assign_tier(20.0) == "C"
        assert assign_tier(24.9) == "C"

    def test_below_c(self):
        assert assign_tier(19.9) == ""
        assert assign_tier(10.0) == ""


# ── Integration: composite with acceleration ──────────────────


def _make_snapshot(coin="TEST", pct_24h=20.0, volume_24h=2_000_000.0,
                   funding_rate=0.005, open_interest=500_000.0, price=100.0):
    return {
        "coin": coin, "price": price,
        "prev_day_price": price / (1 + pct_24h / 100),
        "pct_24h": pct_24h, "volume_24h": volume_24h,
        "funding_rate": funding_rate, "open_interest": open_interest,
    }


class TestCompositeWithAccel:
    def test_accel_boosts_score(self):
        """With accel data, composite should be higher than without."""
        snaps = [_make_snapshot(coin="A")]
        rc = {"A": {"ret_1h": 5.0, "ret_4h": 4.0}}  # accel = 5 - 1 = 4 -> 1.0

        with_accel = score_short_candidates(
            snaps, min_score=0, w_pump=40, w_funding=30, w_oi=20, w_accel=10,
            returns_cache=rc)
        without_accel = score_short_candidates(
            snaps, min_score=0, w_pump=40, w_funding=30, w_oi=20, w_accel=0)

        assert len(with_accel) == 1
        assert len(without_accel) == 1
        assert with_accel[0].composite > without_accel[0].composite
        assert with_accel[0].accel_score > 0

    def test_no_returns_cache_zero_accel(self):
        """Without returns_cache, accel_score should be 0."""
        snaps = [_make_snapshot()]
        result = score_short_candidates(
            snaps, min_score=0, w_pump=40, w_funding=30, w_oi=20, w_accel=10)
        assert result[0].accel_score == 0.0

    def test_tier_assigned(self):
        """Candidates should have tier assigned."""
        snaps = [_make_snapshot(pct_24h=15.0)]
        result = score_short_candidates(snaps, min_score=0)
        assert result[0].tier in ("A", "B", "C", "")

    def test_liquidity_tiebreaker(self):
        """Two candidates with same composite: higher liquidity ranked first."""
        snaps = [
            _make_snapshot(coin="LOW_LIQ", volume_24h=1_100_000, open_interest=100_000),
            _make_snapshot(coin="HIGH_LIQ", volume_24h=500_000_000, open_interest=50_000_000),
        ]
        result = score_short_candidates(snaps, min_score=0)
        # Both have same pct_24h/funding/pump params
        # HIGH_LIQ should have higher liquidity_score
        high = next(c for c in result if c.coin == "HIGH_LIQ")
        low = next(c for c in result if c.coin == "LOW_LIQ")
        assert high.liquidity_score > low.liquidity_score


# ── Percentile integration ────────────────────────────────────


class TestPercentileIntegration:
    def test_pump_percentile_reduces_candidates(self):
        """pump_percentile_keep=0.30 on 15 candidates drops some."""
        snaps = [_make_snapshot(coin=f"C{i}", pct_24h=8.0 + i * 2)
                 for i in range(15)]
        all_result = score_short_candidates(
            snaps, min_score=0, pump_percentile_keep=1.0)
        filtered = score_short_candidates(
            snaps, min_score=0, pump_percentile_keep=0.30)
        assert len(filtered) < len(all_result)
        assert len(filtered) > 0

    def test_small_sample_no_filter(self):
        """< 10 candidates: percentile filter is skipped."""
        snaps = [_make_snapshot(coin=f"C{i}", pct_24h=10.0 + i)
                 for i in range(5)]
        all_result = score_short_candidates(
            snaps, min_score=0, pump_percentile_keep=0.30)
        no_filter = score_short_candidates(
            snaps, min_score=0, pump_percentile_keep=1.0)
        assert len(all_result) == len(no_filter)

    def test_score_percentile_mode(self):
        """score percentile mode filters by composite."""
        snaps = [_make_snapshot(coin=f"C{i}", pct_24h=8.0 + i * 3,
                                funding_rate=0.001 * (i + 1))
                 for i in range(15)]
        filtered = score_short_candidates(
            snaps, min_score=0,
            score_percentile_keep=0.40,
            percentile_mode="score")
        all_result = score_short_candidates(snaps, min_score=0)
        assert len(filtered) < len(all_result)
