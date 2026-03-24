"""Tests for Phase 7: Token Personality Memory."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.token_memory import (
    TokenPersonality,
    _classify_personality,
    _compute_cluster_sensitivity,
    _compute_confidence,
    _compute_mean_reversion,
    _compute_regime_extremes,
    _compute_reversal_speed,
    _compute_single,
    _compute_trend_follow,
    clear_cache,
    compute_conviction_adjustment,
    compute_token_memory,
    format_personality_line,
    get_personality,
)


def _mock_cursor():
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


def _make_tp(**overrides):
    """Helper to create a TokenPersonality with sensible defaults."""
    defaults = dict(
        symbol="TEST", sample_size=20,
        sample_size_clustered=0, sample_size_unclustered=20,
        win_1h=0.5, win_4h=0.5, win_24h=0.5,
        avg_ret_1h=0, avg_ret_4h=0, avg_ret_24h=0,
        avg_mfe_24h=5, avg_mae_24h=-5,
        trend_follow_score=50, mean_reversion_score=50,
        reversal_speed_score=50, cluster_sensitivity=0,
        best_regime=None, worst_regime=None,
        confidence_score=60, personality_label="mixed",
    )
    defaults.update(overrides)
    return TokenPersonality(**defaults)


# ── Trend Follow Score ────────────────────────────────────────


class TestTrendFollow:
    def test_all_positive_returns_high_score(self):
        ret_4h = [3.0, 2.0, 5.0, 1.0]
        ret_24h = [8.0, 6.0, 10.0, 4.0]
        score = _compute_trend_follow(ret_4h, ret_24h)
        assert score >= 90.0  # all positive = strong trend follower

    def test_all_negative_returns_low_score(self):
        ret_4h = [-3.0, -2.0, -5.0, -1.0]
        ret_24h = [-8.0, -6.0, -10.0, -4.0]
        score = _compute_trend_follow(ret_4h, ret_24h)
        assert score <= 10.0  # all negative = weak trend follower

    def test_mixed_returns_moderate(self):
        ret_4h = [3.0, -2.0, 5.0, -1.0]
        ret_24h = [8.0, -6.0, 10.0, -4.0]
        score = _compute_trend_follow(ret_4h, ret_24h)
        assert 30.0 <= score <= 70.0

    def test_empty_returns_neutral(self):
        assert _compute_trend_follow([], []) == 50.0

    def test_score_bounded(self):
        score = _compute_trend_follow([100] * 10, [100] * 10)
        assert 0.0 <= score <= 100.0


# ── Mean Reversion Score ──────────────────────────────────────


class TestMeanReversion:
    def test_all_negative_high_score(self):
        ret_1h = [-1.0, -2.0, -0.5]
        ret_4h = [-3.0, -2.0, -4.0]
        ret_24h = [-8.0, -5.0, -10.0]
        score = _compute_mean_reversion(ret_1h, ret_4h, ret_24h)
        assert score >= 90.0

    def test_all_positive_low_score(self):
        ret_1h = [1.0, 2.0, 0.5]
        ret_4h = [3.0, 2.0, 4.0]
        ret_24h = [8.0, 5.0, 10.0]
        score = _compute_mean_reversion(ret_1h, ret_4h, ret_24h)
        assert score <= 10.0

    def test_empty_24h_neutral(self):
        assert _compute_mean_reversion([1.0], [1.0], []) == 50.0


# ── Reversal Speed ────────────────────────────────────────────


class TestReversalSpeed:
    def test_frequent_fast_reversals_high(self):
        ret_1h = [-3.0, -2.5, -4.0, -1.0, -2.0]
        score = _compute_reversal_speed(ret_1h, [])
        assert score >= 60.0

    def test_no_negative_1h_low(self):
        ret_1h = [1.0, 2.0, 0.5, 3.0]
        score = _compute_reversal_speed(ret_1h, [])
        assert score <= 25.0

    def test_empty_neutral(self):
        assert _compute_reversal_speed([], []) == 50.0


# ── Cluster Sensitivity ──────────────────────────────────────


class TestClusterSensitivity:
    def test_cluster_helps_positive(self):
        clustered = [(None, None, None, -8.0, None, None, "ai")]
        unclustered = [(None, None, None, -2.0, None, None, None)]
        sens = _compute_cluster_sensitivity(clustered, unclustered)
        assert sens > 0

    def test_cluster_hurts_negative(self):
        clustered = [(None, None, None, 5.0, None, None, "meme")]
        unclustered = [(None, None, None, -3.0, None, None, None)]
        sens = _compute_cluster_sensitivity(clustered, unclustered)
        assert sens < 0

    def test_no_cluster_data_zero(self):
        assert _compute_cluster_sensitivity([], [(None,) * 7]) == 0.0
        assert _compute_cluster_sensitivity([(None,) * 7], []) == 0.0


# ── Regime Extremes ───────────────────────────────────────────


class TestRegimeExtremes:
    def test_best_worst_with_data(self):
        regime_stats = {
            "UPTREND_EXPANSION": {"avg_24h": 5.0, "n": 10},
            "DOWNTREND_EXPANSION": {"avg_24h": -8.0, "n": 5},
            "CHOP_EXPANSION": {"avg_24h": -2.0, "n": 7},
        }
        best, worst = _compute_regime_extremes(regime_stats, min_sample=3)
        assert best == "DOWNTREND_EXPANSION"
        assert worst == "UPTREND_EXPANSION"

    def test_below_min_sample(self):
        regime_stats = {
            "UPTREND_EXPANSION": {"avg_24h": 5.0, "n": 2},
        }
        best, worst = _compute_regime_extremes(regime_stats, min_sample=3)
        assert best is None
        assert worst is None

    def test_empty_stats(self):
        best, worst = _compute_regime_extremes({}, 3)
        assert best is None
        assert worst is None


# ── Confidence Score ──────────────────────────────────────────


class TestConfidence:
    def test_large_consistent_sample(self):
        ret_24h = [-5.0] * 50
        score = _compute_confidence(50, ret_24h)
        assert score >= 70.0

    def test_small_sample_lower(self):
        ret_24h = [-5.0] * 5
        small_score = _compute_confidence(5, ret_24h)
        ret_24h_big = [-5.0] * 50
        big_score = _compute_confidence(50, ret_24h_big)
        assert big_score > small_score

    def test_high_variance_lower(self):
        consistent = [-5.0] * 20
        noisy = [-20.0, 15.0, -3.0, 8.0, -1.0] * 4
        score_c = _compute_confidence(20, consistent)
        score_n = _compute_confidence(20, noisy)
        assert score_c > score_n

    def test_bounded(self):
        score = _compute_confidence(1000, [-5.0] * 1000)
        assert 0.0 <= score <= 100.0


# ── Personality Labels ────────────────────────────────────────


class TestClassifyPersonality:
    def test_fade_prone(self):
        label = _classify_personality(
            trend_follow=30, mean_reversion=75, reversal_speed=50,
            cluster_sensitivity=0, confidence=60,
        )
        assert label == "fade-prone"

    def test_fast_reverting(self):
        label = _classify_personality(
            trend_follow=30, mean_reversion=70, reversal_speed=70,
            cluster_sensitivity=0, confidence=60,
        )
        assert label == "fast-reverting"

    def test_trend_following(self):
        label = _classify_personality(
            trend_follow=65, mean_reversion=30, reversal_speed=40,
            cluster_sensitivity=0, confidence=60,
        )
        assert label == "trend-following"

    def test_cluster_sensitive(self):
        label = _classify_personality(
            trend_follow=50, mean_reversion=50, reversal_speed=50,
            cluster_sensitivity=5.0, confidence=50,
        )
        assert label == "cluster-sensitive"

    def test_mixed(self):
        label = _classify_personality(
            trend_follow=50, mean_reversion=50, reversal_speed=50,
            cluster_sensitivity=1.0, confidence=50,
        )
        assert label == "mixed"

    def test_low_confidence_always_mixed(self):
        label = _classify_personality(
            trend_follow=10, mean_reversion=90, reversal_speed=90,
            cluster_sensitivity=10.0, confidence=20,
        )
        assert label == "mixed"


# ── Conviction Adjustment ─────────────────────────────────────


class TestConvictionAdjustment:
    def setup_method(self):
        clear_cache()

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_fade_prone_boosts(self, mock_get):
        mock_get.return_value = _make_tp(
            symbol="MEME", mean_reversion_score=70, trend_follow_score=30,
            confidence_score=65, personality_label="fade-prone",
        )
        adj, reason = compute_conviction_adjustment("dsn", "MEME", max_boost=6.0, max_penalty=6.0)
        assert adj > 0
        assert adj <= 6.0
        assert "fades" in reason

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_trend_following_penalizes(self, mock_get):
        mock_get.return_value = _make_tp(
            symbol="RESOLV", trend_follow_score=80, mean_reversion_score=20,
            confidence_score=70, personality_label="trend-following",
        )
        adj, reason = compute_conviction_adjustment("dsn", "RESOLV", max_boost=6.0, max_penalty=6.0)
        assert adj < 0
        assert adj >= -6.0
        assert "trends" in reason

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_mixed_no_adjustment(self, mock_get):
        mock_get.return_value = _make_tp(personality_label="mixed")
        adj, reason = compute_conviction_adjustment("dsn", "X")
        assert adj == 0.0
        assert reason == ""

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_no_data_no_adjustment(self, mock_get):
        mock_get.return_value = None
        adj, reason = compute_conviction_adjustment("dsn", "UNKNOWN")
        assert adj == 0.0

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_adjustment_bounded(self, mock_get):
        mock_get.return_value = _make_tp(
            sample_size=100, mean_reversion_score=95, trend_follow_score=5,
            confidence_score=95, personality_label="fade-prone",
        )
        adj, _ = compute_conviction_adjustment("dsn", "X", max_boost=6.0, max_penalty=6.0)
        assert adj <= 6.0

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_low_confidence_small_adjustment(self, mock_get):
        mock_get.return_value = _make_tp(
            sample_size=5, mean_reversion_score=80, trend_follow_score=20,
            confidence_score=45, personality_label="fade-prone",
        )
        adj, _ = compute_conviction_adjustment("dsn", "X", max_boost=6.0, max_penalty=6.0)
        assert 0 < adj < 3.0

    # ── Regime-matching tests ─────────────────────────────────

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_best_regime_match_boosts(self, mock_get):
        mock_get.return_value = _make_tp(
            personality_label="fade-prone", mean_reversion_score=70,
            confidence_score=60,
            best_regime="DOWNTREND_EXPANSION", worst_regime="UPTREND_EXPANSION",
        )
        adj_no_regime, _ = compute_conviction_adjustment(
            "dsn", "MEME", max_boost=6.0, max_penalty=6.0, regime_code="",
        )
        adj_best, reason = compute_conviction_adjustment(
            "dsn", "MEME", max_boost=6.0, max_penalty=6.0,
            regime_code="DOWNTREND_EXPANSION",
        )
        assert adj_best > adj_no_regime
        assert "best regime" in reason

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_worst_regime_match_penalizes(self, mock_get):
        mock_get.return_value = _make_tp(
            personality_label="fade-prone", mean_reversion_score=70,
            confidence_score=60,
            best_regime="DOWNTREND_EXPANSION", worst_regime="UPTREND_EXPANSION",
        )
        adj_no_regime, _ = compute_conviction_adjustment(
            "dsn", "MEME", max_boost=6.0, max_penalty=6.0, regime_code="",
        )
        adj_worst, reason = compute_conviction_adjustment(
            "dsn", "MEME", max_boost=6.0, max_penalty=6.0,
            regime_code="UPTREND_EXPANSION",
        )
        assert adj_worst < adj_no_regime
        assert "worst regime" in reason

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_regime_match_for_mixed_label(self, mock_get):
        """Mixed label gets no label-based adj but CAN get regime adj."""
        mock_get.return_value = _make_tp(
            personality_label="mixed", confidence_score=60,
            best_regime="CHOP_EXPANSION", worst_regime="UPTREND_EXPANSION",
        )
        adj, reason = compute_conviction_adjustment(
            "dsn", "X", regime_code="CHOP_EXPANSION",
        )
        assert adj > 0
        assert "best regime" in reason

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_regime_match_same_best_worst_no_adj(self, mock_get):
        """If best==worst regime, no regime adjustment."""
        mock_get.return_value = _make_tp(
            personality_label="fade-prone", mean_reversion_score=70,
            confidence_score=60,
            best_regime="CHOP_EXPANSION", worst_regime="CHOP_EXPANSION",
        )
        adj_with, _ = compute_conviction_adjustment(
            "dsn", "X", max_boost=6.0, regime_code="CHOP_EXPANSION",
        )
        adj_without, _ = compute_conviction_adjustment(
            "dsn", "X", max_boost=6.0, regime_code="",
        )
        assert adj_with == adj_without

    # ── Confidence minimum tests ──────────────────────────────

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_below_confidence_min_no_adjustment(self, mock_get):
        mock_get.return_value = _make_tp(
            personality_label="fade-prone",
            mean_reversion_score=90, confidence_score=35,
        )
        adj, reason = compute_conviction_adjustment(
            "dsn", "X", confidence_min=40.0,
        )
        assert adj == 0.0
        assert reason == ""

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_at_confidence_min_gets_adjustment(self, mock_get):
        mock_get.return_value = _make_tp(
            personality_label="fade-prone",
            mean_reversion_score=80, confidence_score=40.0,
        )
        adj, _ = compute_conviction_adjustment(
            "dsn", "X", confidence_min=40.0,
        )
        assert adj > 0


# ── Format Personality Line ───────────────────────────────────


class TestFormatLine:
    @patch("hltrader.analysis.token_memory.get_personality")
    def test_fade_prone_line(self, mock_get):
        mock_get.return_value = _make_tp(
            symbol="MEME", confidence_score=65, personality_label="fade-prone",
        )
        line = format_personality_line("dsn", "MEME")
        assert "tends to fade" in line
        assert "65" in line

    @patch("hltrader.analysis.token_memory.get_personality")
    def test_no_data_returns_none(self, mock_get):
        mock_get.return_value = None
        assert format_personality_line("dsn", "UNKNOWN") is None


# ── compute_token_memory integration ──────────────────────────


class TestComputeTokenMemory:
    @patch("hltrader.analysis.token_memory.psycopg2")
    def test_db_failure_returns_zeros(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")
        result = compute_token_memory("bad_dsn")
        assert result == {"computed": 0, "skipped": 0, "total_rows": 0}

    @patch("hltrader.analysis.token_memory._upsert_personality")
    @patch("hltrader.analysis.token_memory._fetch_regime_stats")
    @patch("hltrader.analysis.token_memory.psycopg2")
    def test_computes_qualifying_symbols(self, mock_pg, mock_regime, mock_upsert):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_regime.return_value = {}

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        rows = []
        for i in range(5):
            rows.append(("MEME", -1.0 - i, -2.0 - i, -5.0 - i, 2.0, -6.0, None))
        for i in range(2):
            rows.append(("TINY", -0.5, -1.0, -2.0, 1.0, -3.0, None))

        mock_cur.fetchall.side_effect = [rows, []]
        mock_cur.rowcount = 0

        result = compute_token_memory("dsn", min_sample=5)
        assert result["computed"] == 1
        assert result["skipped"] == 1
        assert result["total_rows"] == 7
        mock_upsert.assert_called_once()

    @patch("hltrader.analysis.token_memory._upsert_personality")
    @patch("hltrader.analysis.token_memory._fetch_regime_stats")
    @patch("hltrader.analysis.token_memory.psycopg2")
    def test_skips_all_below_threshold(self, mock_pg, mock_regime, mock_upsert):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_regime.return_value = {}

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        rows = [("TINY", -0.5, -1.0, -2.0, 1.0, -3.0, None)] * 3
        mock_cur.fetchall.side_effect = [rows, []]
        mock_cur.rowcount = 0

        result = compute_token_memory("dsn", min_sample=5)
        assert result["computed"] == 0
        assert result["skipped"] == 1
        mock_upsert.assert_not_called()


# ── _compute_single integration ───────────────────────────────


class TestComputeSingle:
    def test_fade_prone_token(self):
        rows = [("MEME", -2.0, -4.0, -6.0, 2.0, -8.0, None)] * 10
        tp = _compute_single("MEME", rows, {}, 3)
        assert tp.symbol == "MEME"
        assert tp.sample_size == 10
        assert tp.win_24h == 1.0
        assert tp.avg_ret_24h == -6.0
        assert tp.mean_reversion_score > tp.trend_follow_score
        assert tp.personality_label in ("fade-prone", "fast-reverting")

    def test_trend_following_token(self):
        rows = [("RESOLV", 1.0, 3.0, 8.0, 15.0, -2.0, None)] * 10
        tp = _compute_single("RESOLV", rows, {}, 3)
        assert tp.win_24h == 0.0
        assert tp.avg_ret_24h == 8.0
        assert tp.trend_follow_score > tp.mean_reversion_score
        assert tp.personality_label == "trend-following"

    def test_with_regime_data(self):
        rows = [("SYM", -1.0, -2.0, -3.0, 1.0, -4.0, None)] * 10
        regime_stats = {
            "UPTREND_EXPANSION": {"avg_24h": 2.0, "n": 5},
            "DOWNTREND_EXPANSION": {"avg_24h": -8.0, "n": 5},
        }
        tp = _compute_single("SYM", rows, regime_stats, 3)
        assert tp.best_regime == "DOWNTREND_EXPANSION"
        assert tp.worst_regime == "UPTREND_EXPANSION"

    def test_state_serialization_fields(self):
        rows = [("SYM", -1.0, -2.0, -3.0, 1.0, -4.0, None)] * 6
        tp = _compute_single("SYM", rows, {}, 3)
        assert hasattr(tp, "personality_label")
        assert hasattr(tp, "confidence_score")
        assert hasattr(tp, "trend_follow_score")
        assert hasattr(tp, "mean_reversion_score")
        assert hasattr(tp, "reversal_speed_score")
        assert hasattr(tp, "cluster_sensitivity")
        assert 0 <= tp.confidence_score <= 100


# ── Conviction integration in conviction.py ───────────────────


class TestConvictionIntegration:
    """Test that conviction.py properly integrates token memory."""

    @patch("hltrader.analysis.conviction._compute_token_memory_refinement")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_token_memory_boost_applied(self, mock_wr, mock_regime, mock_geo, mock_tm):
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "chop"
        mock_geo.return_value = 50.0

        # Without token memory
        mock_tm.return_value = (0.0, "")
        result_off = compute_conviction(
            symbol="MEME", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=False,
        )

        # With token memory boost
        mock_tm.return_value = (4.0, "token fades (mr=80)")
        result_on = compute_conviction(
            symbol="MEME", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=True,
        )

        assert result_on["conviction"] > result_off["conviction"]
        assert result_on["token_memory_adj"] == 4.0
        assert result_on["components"]["token_memory"] == 4.0

    @patch("hltrader.analysis.conviction._compute_token_memory_refinement")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_token_memory_penalty_applied(self, mock_wr, mock_regime, mock_geo, mock_tm):
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "chop"
        mock_geo.return_value = 50.0
        mock_tm.return_value = (-3.5, "token trends (tf=80)")

        result = compute_conviction(
            symbol="RESOLV", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=True,
        )

        assert result["token_memory_adj"] == -3.5
        assert "token trends" in " ".join(result["reasons"])

    @patch("hltrader.analysis.conviction._compute_token_memory_refinement")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_token_memory_disabled_no_effect(self, mock_wr, mock_regime, mock_geo, mock_tm):
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "chop"
        mock_geo.return_value = 50.0

        result = compute_conviction(
            symbol="MEME", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=False,
        )

        assert result["token_memory_adj"] == 0.0
        mock_tm.assert_not_called()

    @patch("hltrader.analysis.conviction._compute_token_memory_refinement")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_token_memory_passes_regime_code(self, mock_wr, mock_regime, mock_geo, mock_tm):
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "chop"
        mock_geo.return_value = 50.0
        mock_tm.return_value = (2.0, "best regime")

        compute_conviction(
            symbol="MEME", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=True,
            regime_code="CHOP_EXPANSION",
            token_memory_confidence_min=40.0,
        )

        mock_tm.assert_called_once()
        call_kwargs = mock_tm.call_args[1]
        assert call_kwargs["regime_code"] == "CHOP_EXPANSION"

    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_conviction_bounded_after_token_memory(self, mock_wr, mock_regime, mock_geo):
        """Conviction must stay 0-100 even with token memory adjustment."""
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "strong_downtrend"
        mock_geo.return_value = 90.0

        with patch("hltrader.analysis.conviction._compute_token_memory_refinement") as mock_tm:
            mock_tm.return_value = (6.0, "token fades")
            result = compute_conviction(
                symbol="X", alert_type="PENDING",
                composite_score=95, pump_score=90, funding_score=90,
                oi_score=80, accel_score=70, liquidity=0.9,
                alert_timestamp=datetime.now(), pg_dsn="dsn",
                token_memory_enabled=True,
            )
            assert result["conviction"] <= 100.0

    @patch("hltrader.analysis.conviction._compute_geo_component")
    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._fetch_win_rates")
    def test_conviction_returns_token_memory_adj_field(self, mock_wr, mock_regime, mock_geo):
        """compute_conviction always returns token_memory_adj in result."""
        from hltrader.analysis.conviction import compute_conviction, cache_clear
        cache_clear()

        mock_wr.return_value = {"by_type": {}, "by_symbol": {}}
        mock_regime.return_value = "chop"
        mock_geo.return_value = 50.0

        result = compute_conviction(
            symbol="X", alert_type="PENDING",
            composite_score=60, pump_score=50, funding_score=50,
            oi_score=50, accel_score=50, liquidity=0.5,
            alert_timestamp=datetime.now(), pg_dsn="dsn",
            token_memory_enabled=False,
        )

        assert "token_memory_adj" in result
        assert "token_memory" in result["components"]
