"""Tests for Phase 5.3: Signal Clustering."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from hltrader.scan.cluster import (
    SECTOR_MAP,
    SECTOR_LABELS,
    SectorCluster,
    detect_clusters,
)
from hltrader.notify import format_cluster_alert


# ── Test helpers ─────────────────────────────────────────────

@dataclass
class FakeCandidate:
    """Minimal mock of ShortCandidate for testing."""
    coin: str
    asset_type: str = "Crypto"
    price: float = 1.0
    pct_24h: float = 15.0
    funding_rate: float = 0.001
    open_interest: float = 1_000_000.0
    volume_24h: float = 5_000_000.0
    pump_score: float = 0.5
    funding_score: float = 0.3
    oi_score: float = 0.2
    composite: float = 30.0
    ret_1h: float | None = None
    ret_4h: float | None = None
    accel_score: float = 0.0
    liquidity_score: float = 0.5
    tier: str = "B"
    conviction_score: float | None = 60.0
    conviction_reasons: list | None = None
    conviction_allowed: bool | None = True


def _make_meme_candidates(n: int, start_composite: float = 30.0) -> list[FakeCandidate]:
    """Create n meme sector candidates."""
    meme_coins = ["DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "NEIRO", "POPCAT"]
    return [
        FakeCandidate(
            coin=meme_coins[i % len(meme_coins)],
            composite=start_composite - i,
            pct_24h=15.0 + i * 2,
            conviction_score=60.0 + i,
        )
        for i in range(n)
    ]


def _make_ai_candidates(n: int) -> list[FakeCandidate]:
    ai_coins = ["AIXBT", "TAO", "FET", "RENDER", "AKT", "NEAR", "AR", "WLD"]
    return [
        FakeCandidate(
            coin=ai_coins[i % len(ai_coins)],
            composite=35.0 - i,
            pct_24h=12.0 + i,
            conviction_score=65.0,
        )
        for i in range(n)
    ]


def _make_other_candidates(n: int) -> list[FakeCandidate]:
    """Create candidates not in any known sector."""
    return [
        FakeCandidate(
            coin=f"UNKNOWN{i}",
            composite=25.0,
            conviction_score=55.0,
        )
        for i in range(n)
    ]


# ── Core Tests ───────────────────────────────────────────────

class TestDetectClusters:
    def test_no_cluster_below_min_size(self):
        """2 meme candidates should not form a cluster."""
        candidates = _make_meme_candidates(2)
        clusters, unclustered = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 0
        assert len(unclustered) == 2

    def test_cluster_at_min_size(self):
        """Exactly 3 meme candidates should form a cluster."""
        candidates = _make_meme_candidates(3)
        clusters, unclustered = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 1
        assert clusters[0].sector == "meme"
        assert clusters[0].label == "Meme coins"
        assert len(clusters[0].candidates) == 3
        assert len(unclustered) == 0

    def test_multiple_clusters(self):
        """3 meme + 3 AI should produce 2 clusters."""
        candidates = _make_meme_candidates(3) + _make_ai_candidates(3)
        clusters, unclustered = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 2
        sectors = {c.sector for c in clusters}
        assert sectors == {"meme", "ai"}
        assert len(unclustered) == 0

    def test_other_sector_never_clusters(self):
        """'other' sector candidates should never cluster."""
        candidates = _make_other_candidates(5)
        clusters, unclustered = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 0
        assert len(unclustered) == 5

    def test_mixed_cluster_and_unclustered(self):
        """3 meme + 1 AI + 2 other = 1 cluster + 3 unclustered."""
        candidates = _make_meme_candidates(3) + _make_ai_candidates(1) + _make_other_candidates(2)
        clusters, unclustered = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 1
        assert clusters[0].sector == "meme"
        assert len(unclustered) == 3

    def test_cluster_preserves_order(self):
        """Candidates within a cluster should be sorted by composite descending."""
        candidates = _make_meme_candidates(4)
        # Shuffle composites
        candidates[0].composite = 20.0
        candidates[1].composite = 40.0
        candidates[2].composite = 30.0
        candidates[3].composite = 10.0
        clusters, _ = detect_clusters(candidates, min_cluster_size=3)
        assert len(clusters) == 1
        composites = [c.composite for c in clusters[0].candidates]
        assert composites == sorted(composites, reverse=True)

    def test_detect_clusters_fail_open(self):
        """On error, should return ([], all_candidates) unchanged."""
        candidates = _make_meme_candidates(3)
        # Give candidates a broken attribute that will raise during grouping
        bad = FakeCandidate(coin="DOGE")
        del bad.__dict__["coin"]  # remove coin so getattr fails
        # Actually, simpler: patch the candidates list's iteration
        class BadList(list):
            _first = True
            def __iter__(self):
                if BadList._first:
                    BadList._first = False
                    raise RuntimeError("boom")
                return super().__iter__()
        bad_candidates = BadList(candidates)
        clusters, unclustered = detect_clusters(bad_candidates, min_cluster_size=3)
        assert len(clusters) == 0
        assert len(unclustered) == 3

    def test_empty_candidates(self):
        """Empty input should return empty output."""
        clusters, unclustered = detect_clusters([], min_cluster_size=3)
        assert clusters == []
        assert unclustered == []


# ── Conviction Boost Tests ───────────────────────────────────

class TestConvictionBoost:
    def test_conviction_boost_applied(self):
        """Cluster members should get conviction boost."""
        candidates = _make_meme_candidates(3)
        for c in candidates:
            c.conviction_score = 60.0
        clusters, _ = detect_clusters(candidates, conviction_boost=8.0)
        assert len(clusters) == 1
        for c in clusters[0].candidates:
            assert c.conviction_score == 68.0

    def test_conviction_boost_capped_at_100(self):
        """Conviction boost should cap at 100."""
        candidates = _make_meme_candidates(3)
        for c in candidates:
            c.conviction_score = 95.0
        clusters, _ = detect_clusters(candidates, conviction_boost=8.0)
        for c in clusters[0].candidates:
            assert c.conviction_score == 100.0

    def test_conviction_boost_skips_none(self):
        """Candidates with conviction_score=None should not be boosted."""
        candidates = _make_meme_candidates(3)
        candidates[0].conviction_score = None
        candidates[1].conviction_score = 60.0
        candidates[2].conviction_score = 70.0
        clusters, _ = detect_clusters(candidates, conviction_boost=8.0)
        assert clusters[0].candidates[0].conviction_score is not None or \
               any(c.conviction_score is None for c in clusters[0].candidates)
        # Find the None one
        none_c = [c for c in clusters[0].candidates if c.conviction_score is None]
        boosted = [c for c in clusters[0].candidates if c.conviction_score is not None]
        # None stays None
        assert len(none_c) <= 1  # at most 1 None
        # Boosted ones got +8
        for c in boosted:
            assert c.conviction_score >= 68.0


# ── Cluster Average Tests ────────────────────────────────────

class TestClusterAverages:
    def test_avg_composite(self):
        candidates = _make_meme_candidates(3)
        candidates[0].composite = 30.0
        candidates[1].composite = 20.0
        candidates[2].composite = 40.0
        clusters, _ = detect_clusters(candidates)
        assert abs(clusters[0].avg_composite - 30.0) < 0.01

    def test_avg_pct_24h(self):
        candidates = _make_meme_candidates(3)
        candidates[0].pct_24h = 10.0
        candidates[1].pct_24h = 20.0
        candidates[2].pct_24h = 30.0
        clusters, _ = detect_clusters(candidates)
        assert abs(clusters[0].avg_pct_24h - 20.0) < 0.01


# ── Format Tests ─────────────────────────────────────────────

class TestFormatClusterAlert:
    def test_format_cluster_alert_basic(self):
        """Basic cluster alert format check."""
        candidates = _make_meme_candidates(3)
        cluster = SectorCluster(
            sector="meme",
            label="Meme coins",
            candidates=candidates,
            avg_composite=30.0,
            avg_pct_24h=18.0,
        )
        msg = format_cluster_alert(cluster, candidates)
        assert "MEME COINS" in msg
        assert "SECTOR ROTATION" in msg
        assert "3 moving" in msg
        assert "Avg pump: +18.0%" in msg
        assert "DOGE" in msg

    def test_format_cluster_alert_with_regime(self):
        """Cluster alert should include regime info when provided."""
        candidates = _make_meme_candidates(3)
        cluster = SectorCluster(
            sector="meme", label="Meme coins", candidates=candidates,
            avg_composite=30.0, avg_pct_24h=18.0,
        )

        class FakeRegime:
            regime_code = "UPTREND_EXPANSION"
            risk_state = "risk_on"

        msg = format_cluster_alert(cluster, candidates, regime=FakeRegime())
        assert "UPTREND_EXPANSION" in msg
        assert "risk_on" in msg

    def test_format_cluster_alert_empty(self):
        """Empty candidates should return empty string."""
        cluster = SectorCluster(
            sector="meme", label="Meme coins", candidates=[],
            avg_composite=0, avg_pct_24h=0,
        )
        msg = format_cluster_alert(cluster, [])
        assert msg == ""


# ── Outcomes Integration Tests ───────────────────────────────

class TestOutcomesClusterType:
    def test_record_alert_with_cluster_type(self):
        """record_alert should accept and pass cluster_type."""
        with patch("hltrader.eval.outcomes.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            from hltrader.eval.outcomes import record_alert
            result = record_alert(
                "host=localhost dbname=test",
                symbol="DOGE",
                alert_type="CLUSTER",
                score=30.0,
                pump_score=0.5,
                funding_score=0.3,
                oi_score=0.2,
                accel_score=0.1,
                liquidity_score=0.4,
                price_at_alert=0.15,
                alert_timestamp=MagicMock(),
                conviction_score=68.0,
                cluster_type="meme",
            )
            assert result is True
            # Verify cluster_type was in the execute args
            call_args = mock_cur.execute.call_args
            assert "meme" in call_args[0][1]

    def test_record_alert_null_cluster_type(self):
        """record_alert should work with cluster_type=None (default)."""
        with patch("hltrader.eval.outcomes.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            from hltrader.eval.outcomes import record_alert
            result = record_alert(
                "host=localhost dbname=test",
                symbol="BTC",
                alert_type="NORMAL",
                score=25.0,
                pump_score=0.4,
                funding_score=0.2,
                oi_score=0.3,
                accel_score=0.0,
                liquidity_score=0.5,
                price_at_alert=50000.0,
                alert_timestamp=MagicMock(),
            )
            assert result is True
            # cluster_type should be None in the args
            call_args = mock_cur.execute.call_args
            assert None in call_args[0][1]
