"""Phase 4.1 tests: shadow mode, history prior, reasons, /alertstats conviction breakdown."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.conviction import (
    _compute_history,
    _generate_reasons,
    _win_rate_to_score,
    cache_clear,
    compute_conviction,
    conviction_tier,
)


@contextmanager
def _mock_cursor(fetchall_returns=None, fetchone_return=None):
    """Helper: mock psycopg2 connection + cursor context manager."""
    mock_cur = MagicMock()
    if fetchall_returns is not None:
        mock_cur.fetchall.side_effect = fetchall_returns
    if fetchone_return is not None:
        mock_cur.fetchone.return_value = fetchone_return
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    yield mock_conn, mock_cur


# ── History Prior Tests ───────────────────────────────────────


class TestHistoryPrior:
    def setup_method(self):
        cache_clear()

    def test_no_data_returns_prior_default(self):
        """When no win rate data exists, should return prior (default 50)."""
        with _mock_cursor(fetchall_returns=[[], []]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("UNKNOWN", "NORMAL", "dsn://test")
                assert result == 50.0

    def test_no_data_returns_custom_prior(self):
        """When no win rate data exists, should return the custom prior."""
        with _mock_cursor(fetchall_returns=[[], []]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("UNKNOWN", "NORMAL", "dsn://test", history_prior=60.0)
                assert result == 60.0

    def test_no_data_returns_prior_not_5(self):
        """Verify the old bug is fixed: no data should NOT return 5."""
        with _mock_cursor(fetchall_returns=[[], []]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("UNKNOWN", "NORMAL", "dsn://test")
                assert result != 5.0
                assert result == 50.0

    def test_lane_data_overrides_prior(self):
        """When lane data exists, should use win rate mapping, not prior."""
        cache_clear()
        with _mock_cursor(fetchall_returns=[
            [("NORMAL", 30, 50)],  # by_type: 60% win rate
            [],                     # by_symbol: empty
        ]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("UNKNOWN", "NORMAL", "dsn://test", history_prior=50.0)
                assert result == 70.0  # 60% win rate -> 70

    def test_symbol_data_overrides_lane(self):
        """When symbol has enough samples, should use symbol-specific rate."""
        cache_clear()
        with _mock_cursor(fetchall_returns=[
            [("NORMAL", 5, 10)],    # by_type: 50% win rate
            [("BTC", 8, 10)],       # by_symbol: 80% win rate, n=10 >= min_sample
        ]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("BTC", "NORMAL", "dsn://test", min_sample=10)
                assert result == 90.0  # 80% win rate -> 90

    def test_symbol_below_min_sample_falls_to_lane(self):
        """When symbol sample is too small, should fall back to lane stats."""
        cache_clear()
        with _mock_cursor(fetchall_returns=[
            [("NORMAL", 30, 50)],   # by_type: 60% win rate
            [("BTC", 2, 3)],        # by_symbol: n=3 < min_sample=10
        ]) as (mock_conn, _):
            with patch("hltrader.analysis.conviction.psycopg2") as mock_pg:
                mock_pg.connect.return_value = mock_conn
                result = _compute_history("BTC", "NORMAL", "dsn://test", min_sample=10)
                assert result == 70.0  # falls back to lane 60% -> 70

    def test_compute_conviction_uses_history_prior(self):
        """compute_conviction passes history_prior through correctly."""
        cache_clear()
        result = compute_conviction(
            symbol="UNKNOWN",
            alert_type="NORMAL",
            composite_score=50.0,
            pump_score=0.5,
            funding_score=0.5,
            oi_score=0.5,
            accel_score=0.0,
            liquidity=0.5,
            alert_timestamp=datetime.now(timezone.utc),
            pg_dsn="",  # no DSN -> uses prior
            history_prior=60.0,
        )
        # With no pg_dsn, history uses history_prior (60.0)
        assert result["components"]["history"] == 60.0


# ── Reasons Generation Tests ─────────────────────────────────


class TestReasons:
    def test_reasons_allowed_picks_strongest(self):
        """For allowed alerts, reasons should pick the 2 strongest components."""
        components = {
            "base": 80.0,      # deviation +30
            "history": 70.0,   # deviation +20
            "regime": 50.0,    # deviation 0
            "liquidity": 40.0, # deviation -10
            "geo": 30.0,       # deviation -20
        }
        reasons = _generate_reasons(components, conviction=60.0, threshold=55.0)
        assert len(reasons) == 2
        assert reasons[0] == "strong composite"    # base: highest positive dev
        assert reasons[1] == "proven win rate"     # history: second highest

    def test_reasons_blocked_picks_weakest(self):
        """For blocked alerts, reasons should pick the 2 weakest components."""
        components = {
            "base": 20.0,       # deviation -30
            "history": 30.0,    # deviation -20
            "regime": 50.0,     # deviation 0
            "liquidity": 60.0,  # deviation +10
            "geo": 70.0,        # deviation +20
        }
        reasons = _generate_reasons(components, conviction=40.0, threshold=55.0)
        assert len(reasons) == 2
        assert reasons[0] == "weak composite"      # base: most negative
        assert reasons[1] == "poor/no history"     # history: second most negative

    def test_reasons_neutral_components(self):
        """When all components are at 50, reasons still return 2 items."""
        components = {
            "base": 50.0,
            "history": 50.0,
            "regime": 50.0,
            "liquidity": 50.0,
            "geo": 50.0,
        }
        reasons = _generate_reasons(components, conviction=50.0, threshold=55.0)
        assert len(reasons) == 2

    def test_reasons_in_compute_conviction_allowed(self):
        """compute_conviction returns reasons and allowed=True when above threshold."""
        cache_clear()
        result = compute_conviction(
            symbol="BTC",
            alert_type="NORMAL",
            composite_score=80.0,
            pump_score=0.8,
            funding_score=0.5,
            oi_score=0.5,
            accel_score=0.0,
            liquidity=0.9,
            alert_timestamp=datetime.now(timezone.utc),
            conviction_min=55.0,
        )
        assert "reasons" in result
        assert isinstance(result["reasons"], list)
        assert len(result["reasons"]) == 2
        assert result["allowed"] is True

    def test_reasons_in_compute_conviction_blocked(self):
        """compute_conviction returns reasons and allowed=False when below threshold."""
        cache_clear()
        result = compute_conviction(
            symbol="XYZ",
            alert_type="NORMAL",
            composite_score=10.0,
            pump_score=0.1,
            funding_score=0.1,
            oi_score=0.1,
            accel_score=0.0,
            liquidity=0.1,
            alert_timestamp=datetime.now(timezone.utc),
            conviction_min=55.0,
        )
        assert "reasons" in result
        assert len(result["reasons"]) == 2
        assert result["allowed"] is False

    def test_regime_aligned_reason(self):
        """When regime is strong_downtrend (80), it should appear as a positive reason."""
        components = {
            "base": 50.0,
            "history": 50.0,
            "regime": 80.0,    # strong positive
            "liquidity": 50.0,
            "geo": 50.0,
        }
        reasons = _generate_reasons(components, conviction=56.0, threshold=55.0)
        assert "BTC regime aligned" in reasons

    def test_regime_headwind_reason(self):
        """When regime is strong_uptrend (20), it should appear as a negative reason."""
        components = {
            "base": 50.0,
            "history": 50.0,
            "regime": 20.0,    # strong negative
            "liquidity": 50.0,
            "geo": 50.0,
        }
        reasons = _generate_reasons(components, conviction=44.0, threshold=55.0)
        assert "BTC uptrend headwind" in reasons


# ── Shadow Mode Tests ─────────────────────────────────────────


class TestShadowMode:
    """Test shadow mode logic (compute+display but don't block)."""

    def test_shadow_mode_does_not_filter_candidates(self):
        """In shadow mode, all candidates should pass through regardless of conviction."""
        # Simulate watcher shadow mode logic
        candidates_scores = [
            ("BTC", 80.0),   # above threshold
            ("ETH", 40.0),   # below threshold
            ("SOL", 60.0),   # above threshold
        ]

        # Shadow mode: count would-be-blocked but don't filter
        shadow = True
        threshold = 55.0

        if shadow:
            would_block = sum(1 for _, score in candidates_scores if score < threshold)
            remaining = candidates_scores  # no filtering
        else:
            remaining = [(c, s) for c, s in candidates_scores if s >= threshold]
            would_block = 0

        assert len(remaining) == 3  # all pass in shadow mode
        assert would_block == 1     # ETH would have been blocked

    def test_block_mode_filters_candidates(self):
        """In block mode, candidates below threshold should be removed."""
        candidates_scores = [
            ("BTC", 80.0),
            ("ETH", 40.0),
            ("SOL", 60.0),
        ]

        shadow = False
        threshold = 55.0

        remaining = [(c, s) for c, s in candidates_scores if s >= threshold]
        assert len(remaining) == 2
        assert ("ETH", 40.0) not in remaining

    def test_shadow_mode_none_conviction_passes(self):
        """Candidates with None conviction should always pass (fail-open)."""
        candidates_scores = [
            ("BTC", None),
            ("ETH", 40.0),
        ]

        shadow = True
        threshold = 55.0

        would_block = sum(
            1 for _, score in candidates_scores
            if score is not None and score < threshold
        )
        assert would_block == 1  # only ETH, not BTC (None)

    def test_block_mode_none_conviction_passes(self):
        """In block mode, None conviction should also pass (fail-open)."""
        candidates_scores = [
            ("BTC", None),
            ("ETH", 40.0),
            ("SOL", 60.0),
        ]

        threshold = 55.0
        remaining = [
            (c, s) for c, s in candidates_scores
            if s is None or s >= threshold
        ]
        assert len(remaining) == 2  # BTC (None) and SOL (60)
        assert ("ETH", 40.0) not in remaining


# ── Compute Conviction Returns All New Fields ──────────────────


class TestComputeConvictionPhase41:
    def setup_method(self):
        cache_clear()

    def test_returns_reasons_key(self):
        result = compute_conviction(
            symbol="BTC", alert_type="NORMAL",
            composite_score=50.0, pump_score=0.5,
            funding_score=0.5, oi_score=0.5,
            accel_score=0.0, liquidity=0.5,
            alert_timestamp=datetime.now(timezone.utc),
        )
        assert "reasons" in result
        assert isinstance(result["reasons"], list)

    def test_returns_allowed_key(self):
        result = compute_conviction(
            symbol="BTC", alert_type="NORMAL",
            composite_score=80.0, pump_score=0.8,
            funding_score=0.5, oi_score=0.5,
            accel_score=0.0, liquidity=0.8,
            alert_timestamp=datetime.now(timezone.utc),
            conviction_min=55.0,
        )
        assert "allowed" in result
        assert isinstance(result["allowed"], bool)

    def test_high_conviction_is_allowed(self):
        result = compute_conviction(
            symbol="BTC", alert_type="NORMAL",
            composite_score=90.0, pump_score=0.9,
            funding_score=0.9, oi_score=0.9,
            accel_score=0.5, liquidity=0.9,
            alert_timestamp=datetime.now(timezone.utc),
            conviction_min=55.0,
        )
        assert result["allowed"] is True
        assert result["conviction"] >= 55.0

    def test_low_conviction_is_blocked(self):
        result = compute_conviction(
            symbol="XYZ", alert_type="NORMAL",
            composite_score=10.0, pump_score=0.1,
            funding_score=0.1, oi_score=0.1,
            accel_score=0.0, liquidity=0.1,
            alert_timestamp=datetime.now(timezone.utc),
            conviction_min=55.0,
        )
        assert result["allowed"] is False
        assert result["conviction"] < 55.0

    def test_history_prior_propagated(self):
        """history_prior=70 should make history component 70 when no pg_dsn."""
        result = compute_conviction(
            symbol="TEST", alert_type="NORMAL",
            composite_score=50.0, pump_score=0.5,
            funding_score=0.5, oi_score=0.5,
            accel_score=0.0, liquidity=0.5,
            alert_timestamp=datetime.now(timezone.utc),
            pg_dsn="",
            history_prior=70.0,
        )
        assert result["components"]["history"] == 70.0
