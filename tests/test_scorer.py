"""Tests for hltrader.scan.scorer."""

from __future__ import annotations

import pytest

from hltrader.scan.scorer import (
    ShortCandidate,
    score_short_candidates,
    filter_alert_candidates,
    classify_extreme_candidates,
    compute_pump_score,
    compute_liquidity_cutoff,
    MAX_ALERT_SCORE,
)


def _make_snapshot(
    coin="TEST",
    price=100.0,
    pct_24h=20.0,
    volume_24h=1_000_000.0,
    funding_rate=0.005,
    open_interest=500_000.0,
):
    return {
        "coin": coin,
        "price": price,
        "prev_day_price": price / (1 + pct_24h / 100),
        "pct_24h": pct_24h,
        "volume_24h": volume_24h,
        "funding_rate": funding_rate,
        "open_interest": open_interest,
    }


class TestScoring:
    def test_basic_scoring(self):
        """A coin with +20% pump, 0.5% funding, decent OI should score."""
        snaps = [_make_snapshot()]
        result = score_short_candidates(snaps, min_score=0)
        assert len(result) == 1
        c = result[0]
        assert c.coin == "TEST"
        assert c.composite > 0
        # pump_score = min(20/40, 1) = 0.5 (linear, default exp=1)
        assert c.pump_score == pytest.approx(0.5)
        # funding_score = min(0.005/0.01, 1) = 0.5
        assert c.funding_score == pytest.approx(0.5)

    def test_saturated_scores(self):
        """Scores saturate at max values."""
        snaps = [_make_snapshot(pct_24h=80.0, funding_rate=0.02)]
        result = score_short_candidates(snaps, min_score=0)
        c = result[0]
        assert c.pump_score == 1.0
        assert c.funding_score == 1.0

    def test_max_composite_is_100(self):
        """Max possible composite score is 100."""
        snaps = [
            _make_snapshot(pct_24h=80.0, funding_rate=0.02, open_interest=10_000_000),
        ]
        result = score_short_candidates(snaps, min_score=0)
        assert result[0].composite == pytest.approx(100.0)

    def test_negative_pct_filtered(self):
        """Coins with negative 24h change are excluded."""
        snaps = [_make_snapshot(pct_24h=-10.0)]
        result = score_short_candidates(snaps, min_score=0)
        assert len(result) == 0

    def test_negative_funding_scores_zero(self):
        """Coins with negative funding get funding_score=0 but still qualify."""
        snaps = [_make_snapshot(funding_rate=-0.001)]
        result = score_short_candidates(snaps, min_score=0)
        assert len(result) == 1
        assert result[0].funding_score == 0.0
        # Score comes only from pump + OI
        assert result[0].composite > 0

    def test_min_volume_filter(self):
        """Coins below min volume are excluded."""
        snaps = [_make_snapshot(volume_24h=100)]
        result = score_short_candidates(snaps, min_score=0, min_volume=1_000_000)
        assert len(result) == 0

    def test_min_pct_filter(self):
        """Coins below min pct change are excluded."""
        snaps = [_make_snapshot(pct_24h=2.0)]
        result = score_short_candidates(snaps, min_score=0, min_pct_24h=5.0)
        assert len(result) == 0

    def test_min_score_filter(self):
        """Coins below min composite score are excluded."""
        snaps = [_make_snapshot(pct_24h=5.0, funding_rate=0.0001)]
        result = score_short_candidates(snaps, min_score=80)
        assert len(result) == 0

    def test_sorted_by_composite_descending(self):
        """Results are sorted by composite score, highest first."""
        snaps = [
            _make_snapshot(coin="LOW", pct_24h=6.0, funding_rate=0.001),
            _make_snapshot(coin="HIGH", pct_24h=30.0, funding_rate=0.008),
        ]
        result = score_short_candidates(snaps, min_score=0)
        assert len(result) == 2
        assert result[0].coin == "HIGH"
        assert result[1].coin == "LOW"

    def test_oi_p90_normalization(self):
        """OI score uses 90th percentile across all coins (not just qualifying)."""
        snaps = [
            _make_snapshot(coin=f"C{i}", open_interest=float(i * 100_000))
            for i in range(1, 11)
        ]
        result = score_short_candidates(snaps, min_score=0)
        # The coin with highest OI should have oi_score = 1.0
        top = max(result, key=lambda c: c.open_interest)
        assert top.oi_score == 1.0

    def test_empty_snapshots(self):
        """Empty input returns empty list."""
        assert score_short_candidates([]) == []

    def test_returns_dataclass(self):
        """Result items are ShortCandidate dataclass instances."""
        snaps = [_make_snapshot()]
        result = score_short_candidates(snaps, min_score=0)
        assert isinstance(result[0], ShortCandidate)

    def test_min_oi_filter(self):
        """Coins below min OI are excluded."""
        snaps = [_make_snapshot(open_interest=100)]
        result = score_short_candidates(snaps, min_score=0, min_oi=500_000)
        assert len(result) == 0

    def test_min_oi_zero_passes_all(self):
        """min_oi=0 passes everything (default)."""
        snaps = [_make_snapshot(open_interest=0)]
        # OI=0 still passes when min_oi=0 (the coin may have zero OI but meet other criteria)
        result = score_short_candidates(snaps, min_score=0, min_oi=0)
        assert len(result) == 1


class TestFilterAlertCandidates:
    """Tests for the filter_alert_candidates hard cap."""

    def _candidate(self, coin="TEST", composite=30.0):
        return ShortCandidate(
            coin=coin, asset_type="Crypto", price=100.0,
            pct_24h=10.0, funding_rate=0.005, open_interest=500_000.0,
            volume_24h=1_000_000.0, pump_score=0.25, funding_score=0.5,
            oi_score=0.5, composite=composite,
        )

    def test_below_cap_kept(self):
        """Candidates with score < 45 pass through."""
        candidates = [self._candidate(composite=30.0)]
        result = filter_alert_candidates(candidates)
        assert len(result) == 1

    def test_at_cap_kept(self):
        """Candidates with score == 45 pass through."""
        candidates = [self._candidate(composite=45.0)]
        result = filter_alert_candidates(candidates)
        assert len(result) == 1

    def test_above_cap_filtered(self):
        """Candidates with score > 45 are filtered out."""
        candidates = [self._candidate(composite=46.0)]
        result = filter_alert_candidates(candidates)
        assert len(result) == 0

    def test_mixed_list(self):
        """Only candidates <= 45 survive from a mixed list."""
        candidates = [
            self._candidate(coin="LOW", composite=20.0),
            self._candidate(coin="MID", composite=45.0),
            self._candidate(coin="HIGH", composite=80.0),
        ]
        result = filter_alert_candidates(candidates)
        assert len(result) == 2
        assert {c.coin for c in result} == {"LOW", "MID"}

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert filter_alert_candidates([]) == []

    def test_all_above_cap(self):
        """All candidates above cap returns empty."""
        candidates = [
            self._candidate(coin="A", composite=50.0),
            self._candidate(coin="B", composite=99.0),
        ]
        result = filter_alert_candidates(candidates)
        assert len(result) == 0

    def test_custom_max_score(self):
        """Custom max_score overrides the default."""
        candidates = [self._candidate(composite=30.0)]
        result = filter_alert_candidates(candidates, max_score=25.0)
        assert len(result) == 0

    def test_max_alert_score_is_45(self):
        """The hard-coded constant is 45."""
        assert MAX_ALERT_SCORE == 45

    def test_watcher_integration_flow(self):
        """Simulates the full watcher flow: score -> filter -> only <=45 alerted."""
        snaps = [_make_snapshot(coin=f"C{i}", pct_24h=float(i * 5), funding_rate=0.005)
                 for i in range(1, 15)]
        all_candidates = score_short_candidates(snaps, min_score=0)
        alert_candidates = filter_alert_candidates(all_candidates)
        for c in alert_candidates:
            assert c.composite <= 45, f"{c.coin} has score {c.composite} > 45"


class TestComputePumpScore:
    """Tests for the momentum acceleration pump_score curve."""

    def test_linear_default(self):
        """exp=1.0 gives linear behavior (legacy)."""
        assert compute_pump_score(20.0, cap=40.0, exp=1.0) == pytest.approx(0.5)

    def test_zero_pct(self):
        assert compute_pump_score(0.0) == 0.0

    def test_negative_pct_clamps_to_zero(self):
        assert compute_pump_score(-5.0) == 0.0

    def test_saturates_at_cap(self):
        assert compute_pump_score(50.0, cap=25.0, exp=1.6) == pytest.approx(1.0)

    def test_convex_curve_compresses_small(self):
        """With exp=1.6, small pumps score lower than linear."""
        linear = compute_pump_score(10.0, cap=25.0, exp=1.0)
        convex = compute_pump_score(10.0, cap=25.0, exp=1.6)
        assert convex < linear

    def test_convex_values(self):
        """Specific curve values for exp=1.6, cap=25."""
        # 10% pump: norm = 0.4, score = 0.4^1.6
        score = compute_pump_score(10.0, cap=25.0, exp=1.6)
        expected = 0.4 ** 1.6
        assert score == pytest.approx(expected, rel=1e-6)

        # 25% pump: norm = 1.0, score = 1.0
        assert compute_pump_score(25.0, cap=25.0, exp=1.6) == pytest.approx(1.0)

        # 5% pump: norm = 0.2, score = 0.2^1.6
        score5 = compute_pump_score(5.0, cap=25.0, exp=1.6)
        expected5 = 0.2 ** 1.6
        assert score5 == pytest.approx(expected5, rel=1e-6)


class TestComputeLiquidityCutoff:
    """Tests for dynamic percentile cutoff."""

    def test_deterministic_array(self):
        """Known array: P30 cutoff for keep=0.70."""
        values = [100.0, 200.0, 300.0, 400.0, 500.0,
                  600.0, 700.0, 800.0, 900.0, 1000.0]
        cutoff = compute_liquidity_cutoff(values, keep_pct=0.70)
        # drop 30%, idx = floor(10 * 0.30) = 3 → values[3] = 400
        assert cutoff == 400.0

    def test_keep_100_pct_returns_zero(self):
        """keep_pct=1.0 means keep all, cutoff is 0."""
        values = [100.0, 200.0, 300.0]
        assert compute_liquidity_cutoff(values, keep_pct=1.0) == 0.0

    def test_single_value_returns_zero(self):
        """Fewer than 2 values → cutoff = 0."""
        assert compute_liquidity_cutoff([100.0], keep_pct=0.70) == 0.0

    def test_empty_returns_zero(self):
        assert compute_liquidity_cutoff([], keep_pct=0.70) == 0.0

    def test_all_zeros_returns_zero(self):
        """All zero values → no positive values → cutoff = 0."""
        assert compute_liquidity_cutoff([0.0, 0.0, 0.0], keep_pct=0.70) == 0.0

    def test_keep_50_pct(self):
        """Keep top 50%: drop bottom 50%."""
        values = [10.0, 20.0, 30.0, 40.0]
        cutoff = compute_liquidity_cutoff(values, keep_pct=0.50)
        # drop 50%, idx = floor(4 * 0.5) = 2 → sorted[2] = 30
        assert cutoff == 30.0

    def test_ignores_zero_values(self):
        """Zeros are excluded from the distribution."""
        values = [0.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        cutoff = compute_liquidity_cutoff(values, keep_pct=0.70)
        # 5 positive values: [100, 200, 300, 400, 500]
        # drop 30%, idx = floor(5 * 0.3) = 1 → values[1] = 200
        assert cutoff == 200.0


class TestLiquidityPercentileIntegration:
    """Test the liquidity percentile filter inside score_short_candidates."""

    def test_percentile_reduces_candidates(self):
        """keep=0.50 should drop roughly half the candidates."""
        snaps = [
            _make_snapshot(coin=f"C{i}", open_interest=float(i * 100_000))
            for i in range(1, 11)  # OI: 100K to 1M
        ]
        all_result = score_short_candidates(snaps, min_score=0, liquidity_percentile_keep=1.0)
        filtered_result = score_short_candidates(snaps, min_score=0, liquidity_percentile_keep=0.50)
        assert len(filtered_result) < len(all_result)
        assert len(filtered_result) > 0

    def test_percentile_1_keeps_all(self):
        """keep=1.0 keeps everything (no filter)."""
        snaps = [_make_snapshot(coin=f"C{i}", open_interest=float(i * 100_000)) for i in range(1, 6)]
        all_result = score_short_candidates(snaps, min_score=0, liquidity_percentile_keep=1.0)
        assert len(all_result) == 5


class TestClassifyExtremeCandidates:
    """Tests for extreme pump classification."""

    def _candidate(self, coin="TEST", composite=30.0, pct_24h=10.0):
        return ShortCandidate(
            coin=coin, asset_type="Crypto", price=100.0,
            pct_24h=pct_24h, funding_rate=0.005, open_interest=500_000.0,
            volume_24h=1_000_000.0, pump_score=0.25, funding_score=0.5,
            oi_score=0.5, composite=composite,
        )

    def test_normal_stays_normal(self):
        """Low score + low pct → normal lane."""
        candidates = [self._candidate(composite=30.0, pct_24h=10.0)]
        normal, extreme = classify_extreme_candidates(candidates)
        assert len(normal) == 1
        assert len(extreme) == 0

    def test_high_score_routes_to_extreme(self):
        """Score > 45 → extreme lane."""
        candidates = [self._candidate(composite=60.0, pct_24h=15.0)]
        normal, extreme = classify_extreme_candidates(candidates, score_threshold=45.0)
        assert len(normal) == 0
        assert len(extreme) == 1
        assert extreme[0].coin == "TEST"

    def test_high_pct_routes_to_extreme(self):
        """pct_24h >= 20 → extreme lane even with low score."""
        candidates = [self._candidate(composite=30.0, pct_24h=25.0)]
        normal, extreme = classify_extreme_candidates(candidates, pct_threshold=20.0)
        assert len(normal) == 0
        assert len(extreme) == 1

    def test_mixed_split(self):
        """Mixed candidates split correctly."""
        candidates = [
            self._candidate(coin="NORMAL", composite=30.0, pct_24h=10.0),
            self._candidate(coin="EXTREME_SCORE", composite=60.0, pct_24h=15.0),
            self._candidate(coin="EXTREME_PCT", composite=25.0, pct_24h=25.0),
        ]
        normal, extreme = classify_extreme_candidates(
            candidates, score_threshold=45.0, pct_threshold=20.0,
        )
        assert len(normal) == 1
        assert normal[0].coin == "NORMAL"
        assert len(extreme) == 2
        assert {c.coin for c in extreme} == {"EXTREME_SCORE", "EXTREME_PCT"}

    def test_empty_input(self):
        normal, extreme = classify_extreme_candidates([])
        assert normal == []
        assert extreme == []

    def test_at_threshold_boundary(self):
        """Score == 45 is normal (not extreme). pct == 20 IS extreme."""
        candidates = [
            self._candidate(coin="AT_SCORE", composite=45.0, pct_24h=10.0),
            self._candidate(coin="AT_PCT", composite=30.0, pct_24h=20.0),
        ]
        normal, extreme = classify_extreme_candidates(
            candidates, score_threshold=45.0, pct_threshold=20.0,
        )
        assert len(normal) == 1
        assert normal[0].coin == "AT_SCORE"
        assert len(extreme) == 1
        assert extreme[0].coin == "AT_PCT"
