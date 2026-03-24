"""Tests for cross-alert de-duplication (hltrader side)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hltrader.alert_xdedupe import check_recent_alert, record_alert_key
from hltrader.notify import format_extreme_pump_batch, format_short_alert_batch
from hltrader.scan.scorer import ShortCandidate


def _make_candidate(**kwargs):
    defaults = dict(
        coin="GRASS",
        asset_type="Crypto",
        price=0.2784,
        pct_24h=28.0,
        funding_rate=0.008,
        open_interest=6_000_000.0,
        volume_24h=1_500_000.0,
        pump_score=0.9,
        funding_score=0.8,
        oi_score=0.5,
        composite=55.0,
    )
    defaults.update(kwargs)
    return ShortCandidate(**defaults)


class TestRecordAlertKey:
    def test_record_inserts_key(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = record_alert_key(
            mock_conn,
            symbol="GRASS",
            direction="PUMP",
            alert_type="EXTREME",
            source="hltrader-watcher",
        )
        assert result is True
        mock_cur.execute.assert_called_once()
        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO alert_cross_dedupe" in sql
        assert "ON CONFLICT" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "GRASS|PUMP|EXTREME"
        assert params[1] == "GRASS"
        assert params[2] == "PUMP"
        assert params[3] == "hltrader-watcher"
        mock_conn.commit.assert_called_once()

    def test_record_handles_db_error(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("connection lost")

        result = record_alert_key(
            mock_conn,
            symbol="GRASS",
            direction="PUMP",
            alert_type="EXTREME",
            source="hltrader-watcher",
        )
        assert result is False  # fail silently

    def test_key_format(self):
        """Verify the key format is {symbol}|{direction}|{alert_type}."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        record_alert_key(
            mock_conn, symbol="BTC", direction="DROP",
            alert_type="HLMOVE", source="hl_explodes",
        )
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "BTC|DROP|HLMOVE"


class TestCheckRecentAlert:
    def test_found_within_window(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        assert check_recent_alert(
            mock_conn, symbol="GRASS", direction="PUMP",
            alert_type="EXTREME", window_minutes=30,
        ) is True

    def test_not_found(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        assert check_recent_alert(
            mock_conn, symbol="GRASS", direction="PUMP",
            alert_type="EXTREME", window_minutes=30,
        ) is False

    def test_db_error_returns_false(self):
        """Fail-open: DB error means allow the alert."""
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("timeout")

        assert check_recent_alert(
            mock_conn, symbol="GRASS", direction="PUMP",
            alert_type="EXTREME", window_minutes=30,
        ) is False

    def test_check_uses_correct_key(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        check_recent_alert(
            mock_conn, symbol="SOL", direction="PUMP",
            alert_type="EXTREME", window_minutes=60,
        )
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "SOL|PUMP|EXTREME"


class TestFormatExtremePumpEnhanced:
    """Tests for the enhanced EXTREME PUMP message with Why + Playbook."""

    def test_why_flagged_both_conditions(self):
        c = _make_candidate(composite=55.0, pct_24h=28.0)
        msg = format_extreme_pump_batch(
            [c], score_threshold=45.0, pct_threshold=20.0,
        )
        assert "Why flagged:" in msg
        assert "score>45" in msg
        assert "pct>=20%" in msg

    def test_why_flagged_score_only(self):
        c = _make_candidate(composite=55.0, pct_24h=15.0)
        msg = format_extreme_pump_batch(
            [c], score_threshold=45.0, pct_threshold=20.0,
        )
        assert "score>45" in msg
        assert "pct>=20%" not in msg

    def test_why_flagged_pct_only(self):
        c = _make_candidate(composite=40.0, pct_24h=25.0)
        msg = format_extreme_pump_batch(
            [c], score_threshold=45.0, pct_threshold=20.0,
        )
        assert "pct>=20%" in msg
        assert "score>45" not in msg

    def test_playbook_present(self):
        c = _make_candidate(composite=55.0, pct_24h=28.0)
        msg = format_extreme_pump_batch(
            [c], score_threshold=45.0, pct_threshold=20.0,
        )
        assert "Playbook:" in msg
        assert "rejection candle" in msg

    def test_risk_warning_still_present(self):
        c = _make_candidate(composite=55.0, pct_24h=28.0)
        msg = format_extreme_pump_batch([c])
        assert "reversal confirmation" in msg

    def test_backward_compat_no_kwargs(self):
        """Calling without new kwargs still works (uses defaults 45/20)."""
        c = _make_candidate(composite=55.0, pct_24h=28.0)
        msg = format_extreme_pump_batch([c])
        assert "EXTREME PUMP WATCH" in msg
        assert "GRASS" in msg
        assert "Why flagged:" in msg

    def test_empty_still_returns_empty(self):
        assert format_extreme_pump_batch([]) == ""


class TestBeforeAfterGRASS:
    """Integration test: score 55 candidate was dropped by normal, now in EXTREME."""

    def test_normal_lane_drops_high_score(self):
        candidate = _make_candidate(coin="GRASS", composite=55.0, pct_24h=28.0)
        normal_msg = format_short_alert_batch([candidate])
        assert normal_msg == ""  # dropped by defense-in-depth

    def test_extreme_lane_shows_high_score(self):
        candidate = _make_candidate(coin="GRASS", composite=55.0, pct_24h=28.0)
        extreme_msg = format_extreme_pump_batch(
            [candidate], score_threshold=45.0, pct_threshold=20.0,
        )
        assert "GRASS" in extreme_msg
        assert "EXTREME PUMP WATCH" in extreme_msg
        assert "55" in extreme_msg
        assert "Why flagged: score>45 + pct>=20%" in extreme_msg
        assert "Playbook:" in extreme_msg
