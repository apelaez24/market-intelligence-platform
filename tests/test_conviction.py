"""Tests for Phase 4: Conviction Score Fusion Layer."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.conviction import (
    _compute_base,
    _compute_ema,
    _compute_geo_component,
    _compute_history,
    _compute_liquidity_component,
    _compute_regime_score,
    _fetch_active_geo_severity,
    _fetch_btc_regime,
    _fetch_win_rates,
    _win_rate_to_score,
    cache_clear,
    compute_conviction,
    conviction_tier,
)


NOW = datetime(2026, 3, 3, 12, 0, 0, tzinfo=timezone.utc)


def _mock_cursor():
    """Create a mock cursor usable as context manager."""
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear conviction cache before each test."""
    cache_clear()
    yield
    cache_clear()


# ── Base Component ────────────────────────────────────────────


class TestBaseComponent:
    def test_normal_score(self):
        assert _compute_base(42.0) == 42.0

    def test_zero(self):
        assert _compute_base(0.0) == 0.0

    def test_clamp_high(self):
        assert _compute_base(150.0) == 100.0

    def test_clamp_low(self):
        assert _compute_base(-10.0) == 0.0


# ── Win Rate Mapping ──────────────────────────────────────────


class TestWinRateToScore:
    def test_high_win_rate(self):
        assert _win_rate_to_score(0.75) == 90.0

    def test_good_win_rate(self):
        assert _win_rate_to_score(0.65) == 70.0

    def test_neutral_win_rate(self):
        assert _win_rate_to_score(0.55) == 50.0

    def test_poor_win_rate(self):
        assert _win_rate_to_score(0.45) == 30.0

    def test_bad_win_rate(self):
        assert _win_rate_to_score(0.35) == 15.0

    def test_terrible_win_rate(self):
        assert _win_rate_to_score(0.20) == 5.0

    def test_exact_boundary_70(self):
        assert _win_rate_to_score(0.70) == 90.0

    def test_exact_boundary_60(self):
        assert _win_rate_to_score(0.60) == 70.0

    def test_exact_boundary_50(self):
        assert _win_rate_to_score(0.50) == 50.0

    def test_exact_boundary_40(self):
        assert _win_rate_to_score(0.40) == 30.0

    def test_exact_boundary_30(self):
        assert _win_rate_to_score(0.30) == 15.0


# ── Historical Edge ───────────────────────────────────────────


class TestHistoricalEdge:
    @patch("hltrader.analysis.conviction.psycopg2")
    def test_symbol_specific_when_enough_samples(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # Return type stats, then symbol stats
        mock_cur.fetchall.side_effect = [
            [("NORMAL", 30, 50)],       # by_type: 60% win rate
            [("BTC", 7, 10)],           # by_symbol: 70% win rate, n=10
        ]

        score = _compute_history("BTC", "NORMAL", "dsn", min_sample=10)
        assert score == 90.0  # 70% win rate -> 90

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_lane_fallback_when_insufficient_samples(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        mock_cur.fetchall.side_effect = [
            [("NORMAL", 30, 50)],       # by_type: 60% win rate
            [("GRASS", 3, 5)],          # by_symbol: n=5 < min_sample
        ]

        score = _compute_history("GRASS", "NORMAL", "dsn", min_sample=10)
        assert score == 70.0  # falls back to lane 60% -> 70

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_no_data_returns_neutral(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.side_effect = [[], []]

        score = _compute_history("UNKNOWN", "NORMAL", "dsn")
        assert score == 50.0

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_db_failure_returns_neutral(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")
        score = _compute_history("BTC", "NORMAL", "bad_dsn")
        assert score == 50.0

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_cache_prevents_duplicate_queries(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.side_effect = [
            [("NORMAL", 30, 50)],
            [("BTC", 7, 10)],
        ]

        _compute_history("BTC", "NORMAL", "dsn")
        _compute_history("BTC", "NORMAL", "dsn")  # should use cache

        # Only 1 connect call (first one populated cache)
        assert mock_pg.connect.call_count == 1


# ── Regime Context ────────────────────────────────────────────


class TestRegimeScore:
    def test_downtrend_boosts_shorts(self):
        assert _compute_regime_score("strong_downtrend") == 80.0

    def test_uptrend_penalizes_shorts(self):
        assert _compute_regime_score("strong_uptrend") == 20.0

    def test_chop_neutral(self):
        assert _compute_regime_score("chop") == 50.0

    def test_unknown_regime_neutral(self):
        assert _compute_regime_score("unknown") == 50.0


class TestEMA:
    def test_basic_ema(self):
        values = list(range(1, 21))  # 1-20
        ema = _compute_ema(values, 5)
        assert len(ema) == 16  # 20 - 5 + 1
        assert ema[0] == 3.0  # SMA of first 5: (1+2+3+4+5)/5

    def test_insufficient_data(self):
        assert _compute_ema([1, 2, 3], 5) == []

    def test_exact_period(self):
        ema = _compute_ema([10, 20, 30], 3)
        assert len(ema) == 1
        assert ema[0] == 20.0  # SMA of all 3


class TestFetchBtcRegime:
    @patch("hltrader.analysis.conviction.psycopg2")
    def test_strong_uptrend(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # Simulate steadily rising prices
        prices = [60000 + i * 100 for i in range(100)]
        mock_cur.fetchall.return_value = [(p,) for p in reversed(prices)]

        regime = _fetch_btc_regime("dsn_cb")
        assert regime == "strong_uptrend"

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_strong_downtrend(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # Simulate steadily falling prices
        prices = [70000 - i * 100 for i in range(100)]
        mock_cur.fetchall.return_value = [(p,) for p in reversed(prices)]

        regime = _fetch_btc_regime("dsn_cb")
        assert regime == "strong_downtrend"

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_chop_sideways(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # Flat prices
        prices = [67000] * 100
        mock_cur.fetchall.return_value = [(p,) for p in reversed(prices)]

        regime = _fetch_btc_regime("dsn_cb")
        assert regime == "chop"

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_insufficient_data_returns_chop(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [(67000,)] * 10  # only 10 candles

        regime = _fetch_btc_regime("dsn_cb")
        assert regime == "chop"

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_db_failure_returns_chop(self, mock_pg):
        mock_pg.connect.side_effect = Exception("timeout")
        regime = _fetch_btc_regime("bad_dsn")
        assert regime == "chop"

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_regime_cached(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        prices = [67000] * 100
        mock_cur.fetchall.return_value = [(p,) for p in reversed(prices)]

        _fetch_btc_regime("dsn_cb")
        _fetch_btc_regime("dsn_cb")  # cached

        assert mock_pg.connect.call_count == 1


# ── Liquidity Component ──────────────────────────────────────


class TestLiquidityComponent:
    def test_high_liquidity(self):
        assert _compute_liquidity_component(0.9) == 90.0

    def test_low_liquidity(self):
        assert _compute_liquidity_component(0.1) == 10.0

    def test_zero_liquidity(self):
        assert _compute_liquidity_component(0.0) == 0.0

    def test_max_liquidity(self):
        assert _compute_liquidity_component(1.0) == 100.0

    def test_clamp_over(self):
        assert _compute_liquidity_component(1.5) == 100.0


# ── Geo Alignment ─────────────────────────────────────────────


class TestGeoComponent:
    @patch("hltrader.analysis.conviction.psycopg2")
    def test_high_severity_with_alignment(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            (100, ["energy", "risk_off", "defense"]),
        ]

        score = _compute_geo_component("dsn")
        assert score > 50.0  # boosted
        assert score <= 90.0

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_no_geo_events_neutral(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        score = _compute_geo_component("dsn")
        assert score == 50.0

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_geo_no_relevant_watchlist(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            (90, ["semiconductors", "tech"]),  # not in _GEO_RELEVANT_WATCHLIST
        ]

        score = _compute_geo_component("dsn")
        assert score == 50.0  # neutral, no alignment

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_geo_db_failure_neutral(self, mock_pg):
        mock_pg.connect.side_effect = Exception("error")
        score = _compute_geo_component("bad_dsn")
        assert score == 50.0

    @patch("hltrader.analysis.conviction.psycopg2")
    def test_geo_cached(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        _compute_geo_component("dsn")
        _compute_geo_component("dsn")  # cached

        assert mock_pg.connect.call_count == 1


# ── Conviction Tier ───────────────────────────────────────────


class TestConvictionTier:
    def test_tier_a(self):
        assert conviction_tier(80.0) == "A"

    def test_tier_a_boundary(self):
        assert conviction_tier(75.0) == "A"

    def test_tier_b(self):
        assert conviction_tier(65.0) == "B"

    def test_tier_b_boundary(self):
        assert conviction_tier(60.0) == "B"

    def test_tier_c(self):
        assert conviction_tier(57.0) == "C"

    def test_tier_c_boundary(self):
        assert conviction_tier(55.0) == "C"

    def test_below_c(self):
        assert conviction_tier(50.0) == ""

    def test_zero(self):
        assert conviction_tier(0.0) == ""


# ── Full Conviction Computation ───────────────────────────────


class TestComputeConviction:
    def test_all_neutral_components(self):
        """With no DB access (empty DSNs), all DB components default to neutral (50)."""
        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=40.0,
            pump_score=0.8,
            funding_score=0.5,
            oi_score=0.3,
            accel_score=0.2,
            liquidity=0.5,
            alert_timestamp=NOW,
        )
        assert "conviction" in result
        assert "tier" in result
        assert "components" in result
        assert "regime" in result
        assert result["regime"] == "chop"
        # base=40, history=50, regime=50, liq=50, geo=50
        # weighted: 40*0.4 + 50*0.2 + 50*0.2 + 50*0.1 + 50*0.1 = 16+10+10+5+5 = 46
        assert result["conviction"] == 46.0

    def test_high_composite_high_liquidity(self):
        result = compute_conviction(
            symbol="ETH",
            alert_type="EXTREME",
            composite_score=80.0,
            pump_score=0.9,
            funding_score=0.8,
            oi_score=0.7,
            accel_score=0.5,
            liquidity=0.9,
            alert_timestamp=NOW,
        )
        # base=80, history=50, regime=50, liq=90, geo=50
        # 80*0.4 + 50*0.2 + 50*0.2 + 90*0.1 + 50*0.1 = 32+10+10+9+5 = 66
        assert result["conviction"] == 66.0
        assert result["tier"] == "B"

    def test_custom_weights(self):
        result = compute_conviction(
            symbol="SOL",
            alert_type="NORMAL",
            composite_score=50.0,
            pump_score=0.5,
            funding_score=0.5,
            oi_score=0.5,
            accel_score=0.5,
            liquidity=0.5,
            alert_timestamp=NOW,
            w_base=100.0,
            w_history=0.0,
            w_regime=0.0,
            w_liquidity=0.0,
            w_geo=0.0,
        )
        # 100% base weight, score=50
        assert result["conviction"] == 50.0

    def test_conviction_clamp(self):
        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=100.0,
            pump_score=1.0,
            funding_score=1.0,
            oi_score=1.0,
            accel_score=1.0,
            liquidity=1.0,
            alert_timestamp=NOW,
        )
        assert 0.0 <= result["conviction"] <= 100.0

    def test_returns_components_breakdown(self):
        result = compute_conviction(
            symbol="TAO",
            alert_type="NORMAL",
            composite_score=35.0,
            pump_score=0.6,
            funding_score=0.4,
            oi_score=0.3,
            accel_score=0.1,
            liquidity=0.7,
            alert_timestamp=NOW,
        )
        components = result["components"]
        assert "base" in components
        assert "history" in components
        assert "regime" in components
        assert "liquidity" in components
        assert "geo" in components
        assert components["base"] == 35.0
        assert components["liquidity"] == 70.0

    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._compute_history")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    def test_downtrend_boosts_conviction(self, mock_geo, mock_hist, mock_regime):
        mock_regime.return_value = "strong_downtrend"
        mock_hist.return_value = 70.0
        mock_geo.return_value = 80.0

        result = compute_conviction(
            symbol="BTC",
            alert_type="EXTREME",
            composite_score=60.0,
            pump_score=0.9,
            funding_score=0.8,
            oi_score=0.5,
            accel_score=0.3,
            liquidity=0.8,
            alert_timestamp=NOW,
            pg_dsn="dsn",
            pg_dsn_coinbase="dsn_cb",
        )
        # regime=strong_downtrend -> score=80
        assert result["components"]["regime"] == 80.0
        assert result["conviction"] > 60.0  # boosted

    @patch("hltrader.analysis.conviction._fetch_btc_regime")
    @patch("hltrader.analysis.conviction._compute_history")
    @patch("hltrader.analysis.conviction._compute_geo_component")
    def test_uptrend_penalizes_conviction(self, mock_geo, mock_hist, mock_regime):
        mock_regime.return_value = "strong_uptrend"
        mock_hist.return_value = 50.0
        mock_geo.return_value = 50.0

        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=40.0,
            pump_score=0.5,
            funding_score=0.3,
            oi_score=0.2,
            accel_score=0.1,
            liquidity=0.5,
            alert_timestamp=NOW,
            pg_dsn="dsn",
            pg_dsn_coinbase="dsn_cb",
        )
        # regime=strong_uptrend -> score=20 (penalty)
        assert result["components"]["regime"] == 20.0


# ── Feature Flag / No-DSN Path ────────────────────────────────


class TestFeatureFlagPath:
    def test_no_pg_dsn_history_neutral(self):
        """When pg_dsn is empty, history defaults to 50."""
        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=40.0,
            pump_score=0.5,
            funding_score=0.3,
            oi_score=0.2,
            accel_score=0.1,
            liquidity=0.5,
            alert_timestamp=NOW,
            pg_dsn="",
        )
        assert result["components"]["history"] == 50.0

    def test_no_coinbase_dsn_regime_chop(self):
        """When pg_dsn_coinbase is empty, regime defaults to chop."""
        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=40.0,
            pump_score=0.5,
            funding_score=0.3,
            oi_score=0.2,
            accel_score=0.1,
            liquidity=0.5,
            alert_timestamp=NOW,
            pg_dsn_coinbase="",
        )
        assert result["regime"] == "chop"
        assert result["components"]["regime"] == 50.0


# ── Cache Behavior ────────────────────────────────────────────


class TestCache:
    def test_cache_clear(self):
        from hltrader.analysis.conviction import _cache, _cache_set
        _cache_set("test_key", "test_value")
        assert _cache.get("test_key") is not None
        cache_clear()
        assert _cache.get("test_key") is None

    def test_cache_expiry(self):
        from hltrader.analysis.conviction import _cache_get, _cache_set
        _cache_set("expire_test", "value", ttl=0)  # immediate expiry
        # The entry is technically expired at time.time() + 0
        # Since time.time() > expires_at is checked, 0 TTL should expire immediately
        import time
        time.sleep(0.01)
        assert _cache_get("expire_test") is None

    def test_cache_hit(self):
        from hltrader.analysis.conviction import _cache_get, _cache_set
        _cache_set("hit_test", {"data": 42}, ttl=300)
        result = _cache_get("hit_test")
        assert result == {"data": 42}
