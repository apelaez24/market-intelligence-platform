"""Tests for Phase 3: Alert outcome tracking and evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from hltrader.eval.outcomes import (
    _COINBASE_SYMBOLS,
    _lookup_mfe_mae,
    _lookup_return,
    _mark_non_coinbase,
    _update_eval,
    _update_eval_24h,
    evaluate_pending,
    get_weekly_stats,
    record_alert,
)


NOW = datetime(2026, 3, 3, 12, 0, 0, tzinfo=timezone.utc)


def _mock_cursor():
    """Create a mock cursor usable as context manager."""
    mock_cur = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cur)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_cur


# ── record_alert ──────────────────────────────────────────────


class TestRecordAlert:
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_record_alert_inserts_row(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        result = record_alert(
            "host=localhost dbname=market_intel",
            symbol="BTC",
            alert_type="EXTREME",
            score=42.0,
            pump_score=0.9,
            funding_score=0.8,
            oi_score=0.5,
            accel_score=0.3,
            liquidity_score=0.7,
            price_at_alert=67000.0,
            alert_timestamp=NOW,
        )
        assert result is True
        mock_cur.execute.assert_called_once()
        sql = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO alert_outcomes" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "BTC"
        assert params[1] == "EXTREME"
        assert params[8] == 67000.0
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_record_alert_db_failure_returns_false(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")
        result = record_alert(
            "bad_dsn",
            symbol="ETH",
            alert_type="NORMAL",
            score=30.0,
            pump_score=0.5,
            funding_score=0.3,
            oi_score=0.2,
            accel_score=0.1,
            liquidity_score=0.4,
            price_at_alert=3500.0,
            alert_timestamp=NOW,
        )
        assert result is False

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_record_alert_commit_failure(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_conn.commit.side_effect = Exception("disk full")

        result = record_alert(
            "dsn",
            symbol="SOL",
            alert_type="NORMAL",
            score=25.0,
            pump_score=0.4,
            funding_score=0.2,
            oi_score=0.3,
            accel_score=0.0,
            liquidity_score=0.5,
            price_at_alert=150.0,
            alert_timestamp=NOW,
        )
        assert result is False
        mock_conn.close.assert_called_once()


# ── _lookup_return ────────────────────────────────────────────


class TestLookupReturn:
    def test_evaluate_1h_return(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        # Price went from 67000 to 66330 => -1.0%
        mock_cur.fetchone.return_value = (66330.0,)

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=1)
        assert ret is not None
        assert abs(ret - (-1.0)) < 0.01

    def test_evaluate_4h_return(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = (65640.0,)  # -2.0%

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=4)
        assert ret is not None
        assert abs(ret - (-2.03)) < 0.1

    def test_evaluate_24h_return(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = (63650.0,)  # -5.0%

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=24)
        assert ret is not None
        assert abs(ret - (-5.0)) < 0.01

    def test_evaluate_no_candle_data_returns_none(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = None

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=1)
        assert ret is None

    def test_evaluate_db_failure_returns_none(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.execute.side_effect = Exception("timeout")

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=1)
        assert ret is None


# ── _lookup_mfe_mae ───────────────────────────────────────────


class TestMfeMae:
    def test_mfe_mae_computation(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        # High=68340 (+2%), Low=65660 (-2%)
        mock_cur.fetchone.return_value = (68340.0, 65660.0)

        mfe, mae = _lookup_mfe_mae(mock_conn, "btcusd_1h", NOW, 67000.0)
        assert mfe is not None
        assert mae is not None
        assert abs(mfe - 2.0) < 0.01
        assert abs(mae - (-2.0)) < 0.01

    def test_mfe_mae_no_data(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = (None, None)

        mfe, mae = _lookup_mfe_mae(mock_conn, "btcusd_1h", NOW, 67000.0)
        assert mfe is None
        assert mae is None

    def test_mfe_mae_db_failure(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.execute.side_effect = Exception("error")

        mfe, mae = _lookup_mfe_mae(mock_conn, "btcusd_1h", NOW, 67000.0)
        assert mfe is None
        assert mae is None


# ── Return % computation ──────────────────────────────────────


class TestReturnPctComputation:
    def test_positive_return(self):
        """Price went up = positive return."""
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = (70000.0,)

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=1)
        assert ret is not None
        assert ret > 0  # price went up

    def test_negative_return(self):
        """Price went down = negative return (good for shorts)."""
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchone.return_value = (64000.0,)

        ret = _lookup_return(mock_conn, "btcusd_1h", NOW, 67000.0, hours=1)
        assert ret is not None
        assert ret < 0  # price went down


# ── evaluate_pending ──────────────────────────────────────────


class TestEvaluatePending:
    @patch("hltrader.eval.outcomes.psycopg2")
    @patch("hltrader.eval.outcomes._hl_lookup_mfe_mae")
    @patch("hltrader.eval.outcomes._hl_lookup_return")
    def test_evaluate_non_coinbase_symbol_uses_hl(self, mock_hl_ret, mock_hl_mfe, mock_pg):
        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # Row with non-coinbase symbol GRASS, old enough for all horizons
        alert_ts = NOW - timedelta(hours=48)
        mock_cur.fetchall.return_value = [
            ("uuid1", "GRASS", 0.25, alert_ts, False, False, False),
        ]

        mock_hl_ret.return_value = -2.0
        mock_hl_mfe.return_value = (1.0, -3.0)

        with patch("hltrader.eval.outcomes.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            summary = evaluate_pending("dsn_mi", "dsn_cb")

        # Should use HL fallback, not skip
        assert summary["skipped"] == 0
        assert summary["evaluated_1h"] == 1
        assert summary["hl_evaluated"] == 3

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_evaluate_batch_size_limit(self, mock_pg):
        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        evaluate_pending("dsn_mi", "dsn_cb", batch_size=50)
        # Verify LIMIT was passed as 50
        sql_call = mock_cur.execute.call_args
        assert sql_call[0][1] == (50,)

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_evaluate_returns_summary_dict(self, mock_pg):
        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        result = evaluate_pending("dsn_mi", "dsn_cb")
        assert "evaluated_1h" in result
        assert "evaluated_4h" in result
        assert "evaluated_24h" in result
        assert "skipped" in result

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_evaluate_1h_when_old_enough(self, mock_pg):
        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        # market_intel cursor for SELECT
        mock_cm_mi, mock_cur_mi = _mock_cursor()
        # market_intel cursor for UPDATE
        mock_cm_upd, mock_cur_upd = _mock_cursor()
        mock_conn.cursor.side_effect = [mock_cm_mi, mock_cm_upd]

        # coinbase cursor
        mock_cm_cb, mock_cur_cb = _mock_cursor()
        mock_cb_conn.cursor.return_value = mock_cm_cb

        alert_ts = NOW - timedelta(hours=2)  # 2h old, should eval 1h
        mock_cur_mi.fetchall.return_value = [
            ("uuid1", "BTC", 67000.0, alert_ts, False, False, False),
        ]
        mock_cur_cb.fetchone.return_value = (66330.0,)  # -1%

        with patch("hltrader.eval.outcomes.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            summary = evaluate_pending("dsn_mi", "dsn_cb")

        assert summary["evaluated_1h"] == 1

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_evaluate_db_failure_safe(self, mock_pg):
        """If market_intel connect fails, return empty summary."""
        mock_pg.connect.side_effect = Exception("connection refused")
        summary = evaluate_pending("bad_dsn", "bad_dsn2")
        assert summary == {"evaluated_1h": 0, "evaluated_4h": 0, "evaluated_24h": 0, "skipped": 0, "hl_evaluated": 0}

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_evaluate_coinbase_failure_uses_hl_fallback(self, mock_pg):
        """If coinbase_data connect fails, HL fallback is used for all symbols."""
        mock_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, Exception("coinbase down")]

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        summary = evaluate_pending("dsn_mi", "bad_dsn_cb")
        # Should not crash; returns empty because no pending rows
        assert summary["evaluated_1h"] == 0
        assert summary["hl_evaluated"] == 0


# ── _update_eval helpers ──────────────────────────────────────


class TestUpdateHelpers:
    def test_update_eval_1h(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm

        _update_eval(mock_conn, "uuid1", "1h", -1.5)
        sql = mock_cur.execute.call_args[0][0]
        assert "eval_1h_return" in sql
        assert "evaluated_1h" in sql
        mock_conn.commit.assert_called_once()

    def test_update_eval_4h(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm

        _update_eval(mock_conn, "uuid1", "4h", -3.2)
        sql = mock_cur.execute.call_args[0][0]
        assert "eval_4h_return" in sql
        assert "evaluated_4h" in sql

    def test_update_eval_24h_with_mfe_mae(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm

        _update_eval_24h(mock_conn, "uuid1", -5.0, 2.0, -7.0)
        sql = mock_cur.execute.call_args[0][0]
        assert "eval_24h_return" in sql
        assert "mfe_24h" in sql
        assert "mae_24h" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params == (-5.0, 2.0, -7.0, "uuid1")

    def test_update_eval_db_failure_rolls_back(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm
        mock_conn.commit.side_effect = Exception("error")

        _update_eval(mock_conn, "uuid1", "1h", -1.0)
        mock_conn.rollback.assert_called_once()

    def test_mark_non_coinbase(self):
        mock_cm, mock_cur = _mock_cursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cm

        _mark_non_coinbase(mock_conn, "uuid1", False, False, False)
        sql = mock_cur.execute.call_args[0][0]
        assert "evaluated_1h = TRUE" in sql
        assert "evaluated_4h = TRUE" in sql
        assert "evaluated_24h = TRUE" in sql
        mock_conn.commit.assert_called_once()


# ── get_weekly_stats ──────────────────────────────────────────


class TestWeeklyStats:
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_weekly_stats_empty_table(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        stats = get_weekly_stats("dsn")
        assert stats["total_alerts"] == 0
        assert stats["by_type"] == []
        assert stats["top_symbols"] == []
        assert stats["worst_symbols"] == []

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_weekly_stats_aggregation(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # First call: per-type aggregation
        # Second call: top symbols
        # Third call: worst symbols
        mock_cur.fetchall.side_effect = [
            [
                ("NORMAL", 42, 26, 42, 24, 42, 23, 42, -1.2, -3.4),
                ("EXTREME", 8, 6, 8, 5, 7, 5, 7, -2.8, -6.1),
            ],
            [
                ("BTC", -8.2, 5),
                ("ETH", -5.1, 3),
            ],
            [
                ("AIXBT", 12.3, 2),
            ],
        ]

        stats = get_weekly_stats("dsn")
        assert stats["total_alerts"] == 50
        assert len(stats["by_type"]) == 2
        assert stats["by_type"][0]["alert_type"] == "NORMAL"
        assert stats["by_type"][0]["total"] == 42
        assert stats["by_type"][0]["wins_1h"] == 26
        assert stats["by_type"][0]["avg_1h"] == -1.2
        assert stats["by_type"][1]["alert_type"] == "EXTREME"
        assert len(stats["top_symbols"]) == 2
        assert stats["top_symbols"][0]["symbol"] == "BTC"
        assert stats["top_symbols"][0]["avg_ret"] == -8.2
        assert len(stats["worst_symbols"]) == 1
        assert stats["worst_symbols"][0]["symbol"] == "AIXBT"

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_weekly_stats_win_rate_calculation(self, mock_pg):
        """Win = price went down (eval_Xh_return < 0) for short alerts."""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        # 10 alerts, 7 wins at 1h (70% win rate)
        mock_cur.fetchall.side_effect = [
            [("NORMAL", 10, 7, 10, 6, 10, 5, 10, -0.8, -2.1)],
            [],
            [],
        ]

        stats = get_weekly_stats("dsn")
        t = stats["by_type"][0]
        assert t["wins_1h"] == 7
        assert t["eval_1h"] == 10
        # Win rate = 7/10 = 70%

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_weekly_stats_top_worst_symbols(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm

        mock_cur.fetchall.side_effect = [
            [("NORMAL", 20, 12, 20, 10, 20, 10, 20, -1.0, -2.0)],
            [("SOL", -10.5, 4), ("BTC", -7.2, 6)],
            [("GRASS", 15.0, 3), ("MEME", 8.5, 2)],
        ]

        stats = get_weekly_stats("dsn")
        assert stats["top_symbols"][0]["symbol"] == "SOL"
        assert stats["top_symbols"][0]["avg_ret"] == -10.5
        assert stats["worst_symbols"][0]["symbol"] == "GRASS"
        assert stats["worst_symbols"][0]["avg_ret"] == 15.0

    @patch("hltrader.eval.outcomes.psycopg2")
    def test_weekly_stats_db_failure(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")
        stats = get_weekly_stats("bad_dsn")
        assert stats["total_alerts"] == 0
        assert stats["by_type"] == []


# ── Feature flag ──────────────────────────────────────────────


class TestFeatureFlag:
    @patch("hltrader.eval.outcomes.psycopg2")
    def test_feature_flag_disabled_evaluator_noop(self, mock_pg):
        """When feature flag is off, evaluate_pending should still work
        (it's the caller's responsibility to check the flag)."""
        mock_conn = MagicMock()
        mock_cb_conn = MagicMock()
        mock_pg.connect.side_effect = [mock_conn, mock_cb_conn]

        mock_cm, mock_cur = _mock_cursor()
        mock_conn.cursor.return_value = mock_cm
        mock_cur.fetchall.return_value = []

        # The function itself doesn't check the flag — that's the CLI's job
        summary = evaluate_pending("dsn_mi", "dsn_cb")
        assert summary == {"evaluated_1h": 0, "evaluated_4h": 0, "evaluated_24h": 0, "skipped": 0, "hl_evaluated": 0}


# ── Coinbase symbol mapping ──────────────────────────────────


class TestSymbolMapping:
    def test_coinbase_symbols_contains_majors(self):
        assert "BTC" in _COINBASE_SYMBOLS
        assert "ETH" in _COINBASE_SYMBOLS
        assert "SOL" in _COINBASE_SYMBOLS
        assert "TAO" in _COINBASE_SYMBOLS

    def test_coinbase_symbols_table_names(self):
        assert _COINBASE_SYMBOLS["BTC"] == "btcusd_1h"
        assert _COINBASE_SYMBOLS["ETH"] == "ethusd_1h"
        assert _COINBASE_SYMBOLS["SOL"] == "solusd_1h"
        assert _COINBASE_SYMBOLS["TAO"] == "taousd_1h"

    def test_non_coinbase_symbol_not_in_map(self):
        assert "GRASS" not in _COINBASE_SYMBOLS
        assert "AIXBT" not in _COINBASE_SYMBOLS
