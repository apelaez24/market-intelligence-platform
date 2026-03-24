"""Tests for Phase 6.5: HL candle evaluation fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from hltrader.eval.outcomes import (
    _COINBASE_SYMBOLS,
    _clear_hl_cache,
    _fetch_hl_candles,
    _find_candle_close_at,
    _hl_lookup_mfe_mae,
    _hl_lookup_return,
    _lookup_return,
    _update_eval,
    evaluate_pending,
    reset_null_evals,
    CANDLE_TOLERANCE_MINUTES,
    MAX_CACHE_ENTRIES,
)


NOW = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
ALERT_TS = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
ALERT_MS = int(ALERT_TS.timestamp() * 1000)


def _mock_cursor():
    """Create a mock cursor usable as context manager."""
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


def _make_candles(start_ms: int, n: int = 25, interval_ms: int = 3_600_000,
                  base_price: float = 100.0, drift: float = -0.5):
    """Generate n candles starting from start_ms with slight price drift."""
    candles = []
    for i in range(n):
        price = base_price + i * drift
        candles.append({
            "t": start_ms + i * interval_ms,
            "o": str(price),
            "h": str(price + 2.0),
            "l": str(price - 1.5),
            "c": str(price + 0.5),
            "v": "1000",
        })
    return candles


# ── _find_candle_close_at ─────────────────────────────────────


class TestFindCandleCloseAt:
    def test_exact_match(self):
        candles = [{"t": 1000, "c": "50.0"}, {"t": 2000, "c": "60.0"}]
        result = _find_candle_close_at(candles, 2000, 100)
        assert result == 60.0

    def test_within_tolerance(self):
        candles = [{"t": 1000, "c": "50.0"}, {"t": 2000, "c": "60.0"}]
        result = _find_candle_close_at(candles, 1900, 200)
        assert result == 60.0

    def test_outside_tolerance(self):
        candles = [{"t": 1000, "c": "50.0"}, {"t": 5000, "c": "60.0"}]
        result = _find_candle_close_at(candles, 3000, 100)
        assert result is None

    def test_picks_nearest(self):
        candles = [
            {"t": 1000, "c": "50.0"},
            {"t": 2000, "c": "60.0"},
            {"t": 3000, "c": "70.0"},
        ]
        result = _find_candle_close_at(candles, 2100, 500)
        assert result == 60.0


# ── _hl_lookup_return ─────────────────────────────────────────


class TestHlLookupReturn:
    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_1h_return(self, mock_fetch):
        alert_price = 100.0
        # Candle at +1h has close = 98.0 → -2% return
        candles = _make_candles(ALERT_MS, n=3, base_price=100.0, drift=0.0)
        # Set the candle close at +1h to 98.0
        candles[1]["c"] = "98.0"
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("MEME", ALERT_TS, alert_price, hours=1)
        assert ret is not None
        assert abs(ret - (-2.0)) < 0.1

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_no_candles_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        ret = _hl_lookup_return("MEME", ALERT_TS, 100.0, hours=1)
        assert ret is None

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_4h_return(self, mock_fetch):
        candles = _make_candles(ALERT_MS, n=6, base_price=100.0, drift=0.0)
        candles[4]["c"] = "105.0"  # +5% at 4h
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("MEME", ALERT_TS, 100.0, hours=4)
        assert ret is not None
        assert abs(ret - 5.0) < 0.1

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_24h_return(self, mock_fetch):
        candles = _make_candles(ALERT_MS, n=26, base_price=100.0, drift=0.0)
        candles[24]["c"] = "90.0"  # -10% at 24h
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("MEME", ALERT_TS, 100.0, hours=24)
        assert ret is not None
        assert abs(ret - (-10.0)) < 0.1


# ── _hl_lookup_mfe_mae ───────────────────────────────────────


class TestHlLookupMfeMae:
    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_mfe_mae_computed(self, mock_fetch):
        candles = _make_candles(ALERT_MS, n=26, base_price=100.0, drift=0.0)
        # Set some high/low extremes within 24h window
        candles[5]["h"] = "115.0"   # max high: +15%
        candles[10]["l"] = "88.0"   # min low: -12%
        mock_fetch.return_value = candles

        mfe, mae = _hl_lookup_mfe_mae("MEME", ALERT_TS, 100.0)
        assert mfe is not None
        assert mae is not None
        assert mfe > 10.0   # should capture the 115 high
        assert mae < -10.0  # should capture the 88 low

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_no_candles_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        mfe, mae = _hl_lookup_mfe_mae("MEME", ALERT_TS, 100.0)
        assert mfe is None
        assert mae is None

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_excludes_candles_at_alert_time(self, mock_fetch):
        """Candle at exact alert time should NOT be included in MFE/MAE."""
        candles = _make_candles(ALERT_MS, n=26, base_price=100.0, drift=0.0)
        # Set extreme at exact alert time (should be excluded)
        candles[0]["h"] = "200.0"
        candles[0]["l"] = "50.0"
        mock_fetch.return_value = candles

        mfe, mae = _hl_lookup_mfe_mae("MEME", ALERT_TS, 100.0)
        assert mfe is not None
        # MFE should NOT be 100% (from the 200 at alert time)
        assert mfe < 10.0

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_excludes_candles_beyond_24h(self, mock_fetch):
        """Candles beyond 24h window should NOT be included."""
        candles = _make_candles(ALERT_MS, n=30, base_price=100.0, drift=0.0)
        # Set extreme beyond 24h
        candles[25]["h"] = "200.0"
        candles[25]["l"] = "50.0"
        mock_fetch.return_value = candles

        mfe, mae = _hl_lookup_mfe_mae("MEME", ALERT_TS, 100.0)
        assert mfe is not None
        # Should not include the 200/50 from candle 25
        assert mfe < 10.0


# ── _fetch_hl_candles cache ───────────────────────────────────


class TestHlCandleCache:
    def setup_method(self):
        _clear_hl_cache()

    @patch("hltrader.eval.outcomes.requests.post")
    def test_cache_hit(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"t": 1000, "c": "50", "o": "50", "h": "51", "l": "49", "v": "10"}]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # First call
        result1 = _fetch_hl_candles("TEST", "1h", 1000)
        assert result1 is not None
        assert mock_post.call_count == 1

        # Second call with same params should use cache
        result2 = _fetch_hl_candles("TEST", "1h", 1000)
        assert result2 is not None
        assert mock_post.call_count == 1  # No additional API call

    @patch("hltrader.eval.outcomes.requests.post")
    def test_cache_eviction(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        # Fill cache to MAX_CACHE_ENTRIES
        for i in range(MAX_CACHE_ENTRIES + 1):
            mock_resp.json.return_value = [{"t": i * 1000, "c": "50", "o": "50", "h": "51", "l": "49", "v": "10"}]
            mock_post.return_value = mock_resp
            _fetch_hl_candles(f"COIN{i}", "1h", i * 1000)

        # Cache should not exceed MAX_CACHE_ENTRIES
        from hltrader.eval.outcomes import _hl_candle_cache
        assert len(_hl_candle_cache) <= MAX_CACHE_ENTRIES

    @patch("hltrader.eval.outcomes.requests.post")
    def test_api_failure_returns_none(self, mock_post):
        mock_post.side_effect = Exception("connection error")
        result = _fetch_hl_candles("FAIL", "1h", 1000)
        assert result is None


# ── evaluate_pending with HL fallback ─────────────────────────


class TestEvaluatePendingHlFallback:
    @patch("hltrader.eval.outcomes._hl_lookup_mfe_mae")
    @patch("hltrader.eval.outcomes._hl_lookup_return")
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_hl_fallback_for_non_coinbase(self, mock_pg, mock_hl_ret, mock_hl_mfe):
        """Non-Coinbase symbols should use HL fallback instead of being skipped."""
        _clear_hl_cache()

        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        # Alert row: MEME (not in Coinbase), old enough for all horizons
        alert_ts = NOW - timedelta(hours=48)
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            ("id1", "MEME", 0.05, alert_ts, False, False, False),
        ]

        mock_hl_ret.return_value = -3.5
        mock_hl_mfe.return_value = (2.0, -5.0)

        with patch("hltrader.eval.outcomes.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            summary = evaluate_pending("dsn", "cb_dsn")

        assert summary["evaluated_1h"] == 1
        assert summary["evaluated_4h"] == 1
        assert summary["evaluated_24h"] == 1
        assert summary["hl_evaluated"] == 3
        assert summary["skipped"] == 0
        assert mock_hl_ret.call_count == 3  # 1h, 4h, 24h
        assert mock_hl_mfe.call_count == 1  # only 24h

    @patch("hltrader.eval.outcomes._lookup_mfe_mae")
    @patch("hltrader.eval.outcomes._lookup_return")
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_coinbase_path_unchanged(self, mock_pg, mock_cb_ret, mock_cb_mfe):
        """Coinbase symbols should still use the original path."""
        _clear_hl_cache()

        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        alert_ts = NOW - timedelta(hours=48)
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            ("id1", "BTC", 67000.0, alert_ts, False, False, False),
        ]

        mock_cb_ret.return_value = -1.5
        mock_cb_mfe.return_value = (3.0, -2.0)

        with patch("hltrader.eval.outcomes.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            summary = evaluate_pending("dsn", "cb_dsn")

        assert summary["evaluated_1h"] == 1
        assert summary["hl_evaluated"] == 0
        assert mock_cb_ret.call_count == 3  # 1h, 4h, 24h

    @patch("hltrader.eval.outcomes._hl_lookup_return")
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_hl_fallback_when_coinbase_unavailable(self, mock_pg, mock_hl_ret):
        """If coinbase_data DB is down, all symbols use HL fallback."""
        _clear_hl_cache()

        mock_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, Exception("cb down")]

        alert_ts = NOW - timedelta(hours=2)
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = [
            ("id1", "BTC", 67000.0, alert_ts, False, False, False),
        ]

        mock_hl_ret.return_value = -0.5

        with patch("hltrader.eval.outcomes.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            summary = evaluate_pending("dsn", "cb_dsn")

        assert summary["evaluated_1h"] == 1
        assert summary["hl_evaluated"] == 1
        # BTC used HL fallback because coinbase was down
        mock_hl_ret.assert_called_once()


# ── reset_null_evals ──────────────────────────────────────────


class TestResetNullEvals:
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_resets_null_return_rows(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.rowcount = 100

        result = reset_null_evals("dsn")
        assert result["reset_1h"] == 100
        assert result["reset_4h"] == 100
        assert result["reset_24h"] == 100
        assert mock_cur.execute.call_count == 3
        mock_conn.commit.assert_called_once()

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_sql_only_resets_null_returns(self, mock_pg):
        """Verify SQL targets only rows with NULL returns and TRUE flags."""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.rowcount = 0

        reset_null_evals("dsn")

        # Check the SQL for each horizon
        calls = mock_cur.execute.call_args_list
        assert "evaluated_1h = TRUE AND eval_1h_return IS NULL" in calls[0][0][0]
        assert "evaluated_4h = TRUE AND eval_4h_return IS NULL" in calls[1][0][0]
        assert "evaluated_24h = TRUE" in calls[2][0][0]
        assert "eval_24h_return IS NULL" in calls[2][0][0]

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_db_failure_returns_zeros(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")
        result = reset_null_evals("bad_dsn")
        assert result == {"reset_1h": 0, "reset_4h": 0, "reset_24h": 0}


# ── get_weekly_stats with NULL-aware queries ──────────────────


class TestWeeklyStatsNullAware:
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_stats_exclude_null_returns(self, mock_pg):
        """Verify stats queries filter out NULL return rows."""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # First query returns type stats
        mock_cur.fetchall.side_effect = [
            [("EXTREME", 10, 3, 5, 2, 4, 1, 3, -1.5, -2.0)],
            [("BTC", -5.0, 3)],  # top symbols
            [("MEME", 3.0, 5)],  # worst symbols
        ]

        from hltrader.eval.outcomes import get_weekly_stats
        stats = get_weekly_stats("dsn")

        assert stats["total_alerts"] == 10
        assert len(stats["by_type"]) == 1
        # Verify NULL-aware filter in SQL
        sql = mock_cur.execute.call_args_list[0][0][0]
        assert "eval_1h_return IS NOT NULL" in sql


# ── Tolerance behavior ────────────────────────────────────────


class TestToleranceBehavior:
    def test_tolerance_constant(self):
        """Verify tolerance is 30 minutes."""
        assert CANDLE_TOLERANCE_MINUTES == 30

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_return_with_offset_candle(self, mock_fetch):
        """Candle 20 min after target should still be used."""
        candles = _make_candles(ALERT_MS, n=3, base_price=100.0, drift=0.0)
        # Offset the 1h candle by 20 minutes
        candles[1]["t"] = ALERT_MS + 3_600_000 + 20 * 60_000
        candles[1]["c"] = "95.0"
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("TEST", ALERT_TS, 100.0, hours=1)
        assert ret is not None
        assert abs(ret - (-5.0)) < 0.1

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_return_with_candle_too_far(self, mock_fetch):
        """Candle 45 min from target should NOT be used (beyond 30min tolerance)."""
        # Only one candle, 45 min after 1h target
        candles = [{"t": ALERT_MS + 3_600_000 + 45 * 60_000,
                    "c": "95.0", "o": "100", "h": "101", "l": "94", "v": "10"}]
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("TEST", ALERT_TS, 100.0, hours=1)
        assert ret is None


# ── Edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_zero_price_at_alert(self, mock_fetch):
        """Division by zero protection when price_at_alert is 0."""
        candles = _make_candles(ALERT_MS, n=3)
        mock_fetch.return_value = candles

        ret = _hl_lookup_return("TEST", ALERT_TS, 0.0, hours=1)
        assert ret is None

    @patch("hltrader.eval.outcomes._fetch_hl_candles")
    def test_mfe_mae_zero_price(self, mock_fetch):
        candles = _make_candles(ALERT_MS, n=26)
        mock_fetch.return_value = candles

        mfe, mae = _hl_lookup_mfe_mae("TEST", ALERT_TS, 0.0)
        assert mfe is None
        assert mae is None

    def test_clear_cache(self):
        from hltrader.eval.outcomes import _hl_candle_cache
        _hl_candle_cache[("test", 123)] = [{"t": 1}]
        _clear_hl_cache()
        assert len(_hl_candle_cache) == 0
