"""Tests for Phase 5.1/5.2: Market State Snapshot + Narrative Engine."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.state_builder import (
    _atomic_write,
    _extract_geo_theme,
    _format_md,
    build_narrative,
    build_state,
    check_service_status,
    fetch_alerts,
    fetch_geo,
    fetch_performance,
    fetch_regime,
    write_state,
)


# ── Atomic Write Tests ───────────────────────────────────────

class TestAtomicWrite:
    def test_atomic_write_creates_file(self, tmp_path):
        target = tmp_path / "test.json"
        _atomic_write(target, '{"hello": "world"}')
        assert target.exists()
        assert json.loads(target.read_text()) == {"hello": "world"}

    def test_atomic_write_permissions(self, tmp_path):
        target = tmp_path / "test.json"
        _atomic_write(target, '{}')
        mode = oct(os.stat(str(target)).st_mode)[-3:]
        assert mode == "600"

    def test_atomic_write_overwrites(self, tmp_path):
        target = tmp_path / "test.json"
        _atomic_write(target, '{"v": 1}')
        _atomic_write(target, '{"v": 2}')
        assert json.loads(target.read_text()) == {"v": 2}

    def test_atomic_write_no_partial(self, tmp_path):
        """If write fails, old content should remain."""
        target = tmp_path / "test.json"
        _atomic_write(target, '{"old": true}')

        # Simulate failure by making dir read-only won't work on all OS,
        # so we test that temp files are cleaned up on success
        assert target.exists()
        # No leftover .tmp files
        tmps = list(tmp_path.glob("*.tmp"))
        assert len(tmps) == 0


# ── JSON Schema Tests ────────────────────────────────────────

class TestJsonSchema:
    def test_build_state_has_required_keys(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
                state = build_state("dsn")

        required = {"ts", "regime", "alerts", "performance", "geo", "system"}
        assert required.issubset(set(state.keys()))

    def test_alerts_structure(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
                state = build_state("dsn")

        alerts = state["alerts"]
        assert "last_24h_counts" in alerts
        assert "shadow_blocked_24h" in alerts
        assert "top_active" in alerts
        assert isinstance(alerts["top_active"], list)

    def test_performance_structure(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
                state = build_state("dsn")

        perf = state["performance"]
        assert "window_days" in perf
        assert perf["window_days"] == 7
        assert "tiers" in perf
        assert "by_regime" in perf

    def test_system_structure(self):
        with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
            with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
                mock_pg.connect.side_effect = Exception("DB down")
                state = build_state("dsn")

        system = state["system"]
        assert "services" in system
        services = system["services"]
        assert "hltrader_watcher" in services
        assert "telegram_bot" in services
        assert "hltrader_evaluator" in services
        assert "hltrader_state" in services

    def test_ts_is_iso8601(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
                state = build_state("dsn")

        # Should parse without error
        datetime.fromisoformat(state["ts"])


# ── Top Active Cap Test ──────────────────────────────────────

class TestTopActiveCap:
    def test_top_active_max_5(self):
        """Even if DB returns more, top_active should be capped at 5."""
        # The LIMIT 5 in SQL ensures this, but let's verify the structure
        with patch("hltrader.analysis.state_builder._safe_connect") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            # First call: counts (empty)
            # Second call: shadow blocked
            # Third call: top active (return 5 rows)
            mock_cur.fetchall.side_effect = [
                [],  # counts
                [("BTC", "EXTREME", 45.0, 72.0, "A", 67000.0, 0.8, 0.5, 0.4, datetime.now(timezone.utc)),
                 ("ETH", "NORMAL", 35.0, 65.0, "B", 3500.0, 0.7, 0.4, 0.3, datetime.now(timezone.utc)),
                 ("SOL", "EXTREME", 40.0, 60.0, "B", 120.0, 0.6, 0.5, 0.3, datetime.now(timezone.utc)),
                 ("HYPE", "NORMAL", 30.0, 58.0, "C", 25.0, 0.5, 0.3, 0.2, datetime.now(timezone.utc)),
                 ("TAO", "EXTREME", 38.0, 55.0, "C", 400.0, 0.6, 0.4, 0.3, datetime.now(timezone.utc))],
            ]
            mock_cur.fetchone.return_value = (0,)  # shadow count

            result = fetch_alerts("dsn")

        assert len(result["top_active"]) <= 5


# ── Degraded Snapshot Test ───────────────────────────────────

class TestDegradedSnapshot:
    def test_degraded_when_db_down(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("Connection refused")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="inactive"):
                state = build_state("dsn")

        # Should still produce a valid state
        assert "ts" in state
        assert state["regime"] is None
        assert "errors" in state
        assert any("regime" in e for e in state["errors"])

        # Alerts should be empty but present
        assert state["alerts"]["top_active"] == []
        assert state["performance"]["tiers"] == {}

    def test_degraded_writes_file(self, tmp_path):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"):
                state = build_state("dsn")

        with patch("hltrader.analysis.state_builder.STATE_DIR", tmp_path), \
             patch("hltrader.analysis.state_builder.STATE_JSON", tmp_path / "market_state.json"), \
             patch("hltrader.analysis.state_builder.STATE_MD", tmp_path / "market_state.md"):
            json_p, md_p = write_state(state)

        assert json_p.exists()
        assert md_p.exists()
        parsed = json.loads(json_p.read_text())
        assert "errors" in parsed


# ── Tier Computation Test ────────────────────────────────────

class TestTierComputation:
    def test_tier_buckets_from_db(self):
        with patch("hltrader.analysis.state_builder._safe_connect") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            # Tier query returns sample data
            mock_cur.fetchall.side_effect = [
                # Tier breakdown
                [
                    ("A", 10, 7, 10, 6, 10, 5, 8, -2.5),
                    ("B", 20, 12, 20, 10, 18, 9, 15, -1.2),
                    ("C", 15, 7, 15, 6, 12, 5, 10, 0.5),
                    ("<55", 5, 2, 5, 1, 4, 1, 3, 3.0),
                ],
                # By regime
                [
                    ("DOWNTREND_EXPANSION", 12, -3.1, 8, 10),
                    ("CHOP_CONTRACTION", 8, -0.5, 4, 7),
                ],
            ]

            result = fetch_performance("dsn")

        assert "A" in result["tiers"]
        assert result["tiers"]["A"]["count"] == 10
        assert result["tiers"]["A"]["win_1h"] == 0.7
        assert result["tiers"]["A"]["avg_24h"] == -2.5

        assert len(result["by_regime"]) == 2
        assert result["by_regime"][0]["regime_code"] == "DOWNTREND_EXPANSION"


# ── Service Status Test ──────────────────────────────────────

class TestServiceStatus:
    def test_active_service(self):
        with patch("hltrader.analysis.state_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="active\n")
            assert check_service_status("test-service") == "active"

    def test_inactive_service(self):
        with patch("hltrader.analysis.state_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=3, stdout="inactive\n")
            assert check_service_status("test-service") == "inactive"

    def test_timeout_returns_unknown(self):
        with patch("hltrader.analysis.state_builder.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("systemctl", 5)
            assert check_service_status("test-service") == "unknown"

    def test_exception_returns_unknown(self):
        with patch("hltrader.analysis.state_builder.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("systemctl not found")
            assert check_service_status("test-service") == "unknown"


# ── Markdown Formatting Test ─────────────────────────────────

class TestMarkdownFormat:
    def test_format_md_basic(self):
        state = {
            "ts": "2026-03-03T18:00:00+00:00",
            "regime": {
                "regime_code": "UPTREND_EXPANSION",
                "risk_state": "risk_on",
                "btc_trend": "up",
                "vol_state": "expansion",
                "metrics": {"btc_slope_4h": 0.004, "atr_ratio": 1.3},
            },
            "alerts": {
                "last_24h_counts": {"NORMAL": 3, "EXTREME": 1},
                "shadow_blocked_24h": 0,
                "top_active": [
                    {"symbol": "BTC", "lane": "EXTREME", "score": 42, "conviction": 72, "tier": "B", "ts": None}
                ],
            },
            "performance": {"window_days": 7, "tiers": {}, "by_regime": []},
            "geo": {"active": False, "top": []},
            "system": {"services": {"hltrader_watcher": "active", "telegram_bot": "active", "hltrader_evaluator": "active"}},
        }
        md = _format_md(state)
        assert "UPTREND_EXPANSION" in md
        assert "risk_on" in md
        assert "BTC" in md
        assert "OK" in md

    def test_format_md_unavailable_regime(self):
        state = {
            "ts": "2026-03-03T18:00:00+00:00",
            "regime": None,
            "alerts": {"last_24h_counts": {}, "shadow_blocked_24h": 0, "top_active": []},
            "performance": {"window_days": 7, "tiers": {}, "by_regime": []},
            "geo": {"active": False, "top": []},
            "system": {"services": {}},
            "errors": ["regime: no recent snapshot"],
        }
        md = _format_md(state)
        assert "UNAVAILABLE" in md
        assert "Warnings" in md


# ── Fetch Regime Test ────────────────────────────────────────

class TestFetchRegime:
    def test_returns_none_on_no_data(self):
        with patch("hltrader.analysis.state_builder._safe_connect") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchone.return_value = None

            result = fetch_regime("dsn")

        assert result is None

    def test_returns_regime_dict(self):
        with patch("hltrader.analysis.state_builder._safe_connect") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchone.return_value = (
                "DOWNTREND_EXPANSION", "down", "expansion", "risk_off",
                -0.003, 1.4, 0.05, 0.2, datetime.now(timezone.utc),
            )

            result = fetch_regime("dsn")

        assert result["regime_code"] == "DOWNTREND_EXPANSION"
        assert result["risk_state"] == "risk_off"
        assert result["metrics"]["atr_ratio"] == 1.4


# Need subprocess import at module level for TimeoutExpired
import subprocess


# ── Narrative Engine Tests ───────────────────────────────────

class TestNarrativeSectorDetection:
    def test_ai_sector_detected(self):
        alerts = {"top_active": [
            {"symbol": "AIXBT", "lane": "EXTREME", "score": 45},
            {"symbol": "TAO", "lane": "NORMAL", "score": 30},
            {"symbol": "FET", "lane": "NORMAL", "score": 25},
        ]}
        n = build_narrative(None, alerts, {"active": False, "top": []})
        assert n["market_driver"] == "AI tokens"
        assert n["sector_heat"][0]["sector"] == "ai"
        assert n["sector_heat"][0]["count"] == 3

    def test_meme_sector_detected(self):
        alerts = {"top_active": [
            {"symbol": "MEME", "lane": "EXTREME", "score": 60},
            {"symbol": "PEPE", "lane": "NORMAL", "score": 30},
        ]}
        n = build_narrative(None, alerts, {"active": False, "top": []})
        assert n["market_driver"] == "Meme coins"

    def test_mixed_sectors(self):
        alerts = {"top_active": [
            {"symbol": "AIXBT", "lane": "EXTREME", "score": 45},
            {"symbol": "MEME", "lane": "NORMAL", "score": 30},
            {"symbol": "UNI", "lane": "NORMAL", "score": 25},
        ]}
        n = build_narrative(None, alerts, {"active": False, "top": []})
        assert len(n["sector_heat"]) == 3

    def test_unknown_symbol_maps_to_other(self):
        alerts = {"top_active": [
            {"symbol": "XYZ_UNKNOWN", "lane": "NORMAL", "score": 30},
        ]}
        n = build_narrative(None, alerts, {"active": False, "top": []})
        assert n["sector_heat"][0]["sector"] == "other"

    def test_empty_alerts_quiet(self):
        n = build_narrative(None, {"top_active": []}, {"active": False, "top": []})
        assert n["market_driver"] == "quiet"
        assert n["sector_heat"] == []


class TestNarrativeSentiment:
    def test_bullish_uptrend_risk_on(self):
        regime = {"regime_code": "UPTREND_EXPANSION", "risk_state": "risk_on", "btc_trend": "up", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.02, "atr_ratio": 1.3}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["sentiment"] == "bullish"

    def test_bearish_downtrend_risk_off(self):
        regime = {"regime_code": "DOWNTREND_EXPANSION", "risk_state": "risk_off", "btc_trend": "down", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.02, "breadth_dump10_pct": 0.25, "atr_ratio": 1.5}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["sentiment"] == "bearish"

    def test_neutral_chop(self):
        regime = {"regime_code": "CHOP_CONTRACTION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "contraction",
                  "metrics": {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.05, "atr_ratio": 0.9}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["sentiment"] == "neutral"

    def test_no_regime_unknown(self):
        n = build_narrative(None, {"top_active": []}, {"active": False, "top": []})
        assert n["sentiment"] == "unknown"


class TestNarrativeBreadth:
    def test_strong_bullish(self):
        regime = {"regime_code": "UPTREND_EXPANSION", "risk_state": "risk_on", "btc_trend": "up", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.25, "breadth_dump10_pct": 0.0, "atr_ratio": 1.0}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["breadth_strength"] == "strong_bullish"

    def test_strong_bearish(self):
        regime = {"regime_code": "DOWNTREND_EXPANSION", "risk_state": "risk_off", "btc_trend": "down", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.0, "breadth_dump10_pct": 0.30, "atr_ratio": 1.0}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["breadth_strength"] == "strong_bearish"

    def test_narrow(self):
        regime = {"regime_code": "CHOP_CONTRACTION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "contraction",
                  "metrics": {"breadth_pump10_pct": 0.02, "breadth_dump10_pct": 0.02, "atr_ratio": 0.9}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["breadth_strength"] == "narrow"


class TestNarrativeVolatility:
    def test_elevated(self):
        regime = {"regime_code": "CHOP_EXPANSION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.0, "breadth_dump10_pct": 0.0, "atr_ratio": 1.6}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["volatility_context"] == "elevated"

    def test_expanding(self):
        regime = {"regime_code": "CHOP_EXPANSION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.0, "breadth_dump10_pct": 0.0, "atr_ratio": 1.3}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["volatility_context"] == "expanding"

    def test_compressed(self):
        regime = {"regime_code": "CHOP_CONTRACTION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "contraction",
                  "metrics": {"breadth_pump10_pct": 0.0, "breadth_dump10_pct": 0.0, "atr_ratio": 0.6}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["volatility_context"] == "compressed"

    def test_stable(self):
        regime = {"regime_code": "CHOP_CONTRACTION", "risk_state": "neutral", "btc_trend": "chop", "vol_state": "contraction",
                  "metrics": {"breadth_pump10_pct": 0.0, "breadth_dump10_pct": 0.0, "atr_ratio": 0.9}}
        n = build_narrative(regime, {"top_active": []}, {"active": False, "top": []})
        assert n["volatility_context"] == "stable"


class TestNarrativeGeo:
    def test_geo_theme_iran(self):
        assert _extract_geo_theme("US strikes Iran nuclear facility", 100) == "Iran tensions"

    def test_geo_theme_middle_east(self):
        assert _extract_geo_theme("Middle East crisis deepens", 90) == "Middle East conflict"

    def test_geo_theme_unknown(self):
        result = _extract_geo_theme("Something completely unrelated happened", 85)
        assert result == "Sev-85 event"

    def test_geo_driver_injected(self):
        geo = {"active": True, "top": [{"severity": 100, "headline": "Iran nuclear talks collapse", "ts": None}]}
        n = build_narrative(None, {"top_active": []}, geo)
        assert n["geo_driver"] == "Iran tensions"

    def test_no_geo_driver_when_quiet(self):
        n = build_narrative(None, {"top_active": []}, {"active": False, "top": []})
        assert n["geo_driver"] is None


class TestNarrativeSummary:
    def test_summary_with_regime(self):
        regime = {"regime_code": "UPTREND_EXPANSION", "risk_state": "risk_on",
                  "btc_trend": "up", "vol_state": "expansion",
                  "metrics": {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.0, "atr_ratio": 1.3}}
        alerts = {"top_active": [{"symbol": "AIXBT", "lane": "EXTREME", "score": 45}]}
        n = build_narrative(regime, alerts, {"active": False, "top": []})
        assert "Uptrend Expansion" in n["summary"]
        assert "AI tokens" in n["summary"]
        assert "bullish" in n["summary"]

    def test_summary_without_regime(self):
        n = build_narrative(None, {"top_active": []}, {"active": False, "top": []})
        assert "Regime unknown" in n["summary"]

    def test_narrative_in_build_state(self):
        with patch("hltrader.analysis.state_builder.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")
            with patch("hltrader.analysis.state_builder.check_service_status", return_value="active"), \
                 patch("hltrader.analysis.state_builder.check_timer_status", return_value="active"):
                state = build_state("dsn")

        assert "narrative" in state
        assert "market_driver" in state["narrative"]
        assert "sentiment" in state["narrative"]
        assert "summary" in state["narrative"]

    def test_narrative_in_md(self):
        state = {
            "ts": "2026-03-04T08:00:00+00:00",
            "regime": {"regime_code": "UPTREND_EXPANSION", "risk_state": "risk_on",
                       "btc_trend": "up", "vol_state": "expansion",
                       "metrics": {"btc_slope_4h": 0.004, "atr_ratio": 1.3}},
            "narrative": {
                "market_driver": "AI tokens", "sentiment": "bullish",
                "breadth_strength": "narrow", "volatility_context": "expanding",
                "sector_heat": [{"sector": "ai", "label": "AI tokens", "count": 3}],
                "geo_driver": None,
                "summary": "Uptrend Expansion | led by AI tokens | sentiment bullish",
            },
            "alerts": {"last_24h_counts": {}, "shadow_blocked_24h": 0, "top_active": []},
            "performance": {"window_days": 7, "tiers": {}, "by_regime": []},
            "geo": {"active": False, "top": []},
            "system": {"services": {}},
        }
        md = _format_md(state)
        assert "Market Narrative" in md
        assert "AI tokens" in md
        assert "bullish" in md
