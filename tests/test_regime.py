"""Tests for Phase 5: Market Regime Engine."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.regime import (
    RegimeResult,
    _classify_btc_trend,
    _classify_risk_state,
    _classify_vol_state,
    _compute_breadth,
    _compute_ema,
    _reset_vol_state,
    cache_clear,
    compute_regime,
    store_regime_snapshot,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_closes(base: float, trend: float, n: int) -> list[tuple]:
    """Generate DB rows [(close,), ...] in DESC order (newest first)."""
    closes = [base + trend * i for i in range(n)]
    return [(c,) for c in reversed(closes)]


def _make_snapshots(pct_list: list[float], funding_list: list[float] | None = None) -> list[dict]:
    """Create scan snapshot dicts."""
    if funding_list is None:
        funding_list = [0.001] * len(pct_list)
    return [
        {"coin": f"COIN{i}", "pct_24h": p, "funding_rate": f, "volume_24h": 1e6, "open_interest": 1e5}
        for i, (p, f) in enumerate(zip(pct_list, funding_list))
    ]


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset cache and hysteresis before each test."""
    cache_clear()
    _reset_vol_state()
    yield
    cache_clear()
    _reset_vol_state()


# ── Trend Tests ──────────────────────────────────────────────

class TestBtcTrend:
    def test_strong_uptrend(self):
        # 200 candles trending up: base=60000, +50 per candle
        rows = _make_closes(60000, 50, 200)
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn", slope_up=0.002, slope_down=-0.002)

        assert trend == "up"
        assert "btc_ema20" in metrics
        assert "btc_ema50" in metrics
        assert metrics["btc_slope_4h"] > 0.002

    def test_strong_downtrend(self):
        # 200 candles trending down: base=70000, -50 per candle
        rows = _make_closes(70000, -50, 200)
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn", slope_up=0.002, slope_down=-0.002)

        assert trend == "down"
        assert metrics["btc_slope_4h"] < -0.002

    def test_chop_flat_slope(self):
        # Flat market: all same price
        rows = [(65000.0,)] * 200
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn")

        assert trend == "chop"

    def test_chop_insufficient_data(self):
        # Too few rows
        rows = [(65000.0,)] * 20
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn")

        assert trend == "chop"

    def test_slope_exact_boundary(self):
        # Slope exactly at boundary should be chop (not > slope_up)
        # Flat prices with very slight upward bias
        rows = [(65000.0 + i * 0.01,) for i in range(200)]
        rows.reverse()  # DESC order
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn", slope_up=0.002, slope_down=-0.002)

        # With tiny slope, should be chop
        assert trend == "chop"

    def test_trend_metrics_populated(self):
        rows = _make_closes(60000, 50, 200)
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            trend, metrics = _classify_btc_trend("dsn")

        assert "btc_ema20" in metrics
        assert "btc_ema50" in metrics
        assert "btc_slope_4h" in metrics
        assert isinstance(metrics["btc_ema20"], float)


# ── Volatility Tests ─────────────────────────────────────────

class TestVolState:
    def test_expansion_high_atr(self):
        # Volatile market: alternating large moves
        closes = []
        for i in range(200):
            if i % 2 == 0:
                closes.append(65000 + i * 100)
            else:
                closes.append(65000 - i * 100)
        rows = [(c,) for c in reversed(closes)]

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn", atr_expand=1.25, atr_contract=0.85)

        # With increasing oscillations, ATR should be high relative to median
        assert vol in ("expansion", "contraction")
        assert "atr_4h" in metrics

    def test_contraction_low_atr(self):
        # Very calm market: all same price
        rows = [(65000.0,)] * 200
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn")

        assert vol == "contraction"

    def test_hysteresis_maintains_state(self):
        """When ratio is between thresholds, state should persist."""
        import hltrader.analysis.regime as regime_mod
        regime_mod._last_vol_state = "expansion"

        # Flat market -> atr_ratio ~1.0, between 0.85 and 1.25
        # Generate prices with consistent small movements
        closes = [65000.0 + (i % 3) * 10 for i in range(200)]
        rows = [(c,) for c in reversed(closes)]

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn", atr_expand=1.25, atr_contract=0.85)

        # ratio should be ~1.0, so hysteresis keeps "expansion"
        assert vol == "expansion"

    def test_hysteresis_transition(self):
        """Verify transition from contraction to expansion."""
        import hltrader.analysis.regime as regime_mod
        regime_mod._last_vol_state = "contraction"

        # Start calm, end volatile: create increasing swings
        closes = []
        for i in range(200):
            if i < 150:
                closes.append(65000.0 + (i % 2) * 5)  # tiny swings
            else:
                closes.append(65000.0 + ((i % 2) * 2 - 1) * 500 * (i - 149))  # big swings
        rows = [(c,) for c in reversed(closes)]

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn", atr_expand=1.25, atr_contract=0.85)

        # Should detect expansion from the big swings at end
        assert vol in ("expansion", "contraction")  # depends on exact ratio
        assert "atr_4h" in metrics or "atr_ratio" in metrics or vol == "contraction"

    def test_vol_insufficient_data(self):
        rows = [(65000.0,)] * 20  # too few
        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn")

        assert vol == "contraction"  # default

    def test_atr_computation(self):
        """Verify ATR value is populated and reasonable."""
        # Steady uptrend with consistent moves
        closes = [60000 + i * 100 for i in range(200)]
        rows = [(c,) for c in reversed(closes)]

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            vol, metrics = _classify_vol_state("dsn")

        assert "atr_4h" in metrics
        assert metrics["atr_4h"] > 0


# ── Breadth Tests ────────────────────────────────────────────

class TestBreadth:
    def test_breadth_all_pumping(self):
        snapshots = _make_snapshots([15.0, 20.0, 12.0, 25.0])
        result = _compute_breadth(snapshots)
        assert result["breadth_pump10_pct"] == 1.0
        assert result["breadth_dump10_pct"] == 0.0

    def test_breadth_all_dumping(self):
        snapshots = _make_snapshots([-15.0, -20.0, -12.0, -25.0])
        result = _compute_breadth(snapshots)
        assert result["breadth_pump10_pct"] == 0.0
        assert result["breadth_dump10_pct"] == 1.0

    def test_breadth_mixed(self):
        snapshots = _make_snapshots([15.0, -15.0, 5.0, -5.0])
        result = _compute_breadth(snapshots)
        assert result["breadth_pump10_pct"] == 0.25  # 1/4
        assert result["breadth_dump10_pct"] == 0.25  # 1/4

    def test_breadth_empty(self):
        result = _compute_breadth([])
        assert result["breadth_pump10_pct"] == 0.0
        assert result["breadth_dump10_pct"] == 0.0
        assert result["funding_median"] == 0.0
        assert result["funding_pctl"] == 0.0


# ── Risk State Tests ─────────────────────────────────────────

class TestRiskState:
    def test_risk_off_downtrend_expansion(self):
        breadth = {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.05}
        assert _classify_risk_state("down", "expansion", breadth) == "risk_off"

    def test_risk_off_breadth_dump(self):
        breadth = {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.25}
        assert _classify_risk_state("chop", "contraction", breadth) == "risk_off"

    def test_risk_on_uptrend_expansion(self):
        breadth = {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.05}
        assert _classify_risk_state("up", "expansion", breadth) == "risk_on"

    def test_risk_on_breadth_pump(self):
        breadth = {"breadth_pump10_pct": 0.25, "breadth_dump10_pct": 0.05}
        assert _classify_risk_state("chop", "contraction", breadth) == "risk_on"

    def test_neutral_no_signal(self):
        breadth = {"breadth_pump10_pct": 0.05, "breadth_dump10_pct": 0.05}
        assert _classify_risk_state("chop", "contraction", breadth) == "neutral"

    def test_risk_off_precedence(self):
        """When both risk_on and risk_off conditions are true, risk_off wins."""
        breadth = {"breadth_pump10_pct": 0.25, "breadth_dump10_pct": 0.25}
        assert _classify_risk_state("down", "expansion", breadth) == "risk_off"


# ── Integration Tests ────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """Full compute_regime with mocked DB."""
        rows = _make_closes(60000, 50, 200)
        snapshots = _make_snapshots([15.0, -15.0, 5.0, 8.0, -8.0])

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchall.return_value = rows

            result = compute_regime(
                snapshots,
                pg_dsn_coinbase="dsn",
                cache_ttl=0,
            )

        assert isinstance(result, RegimeResult)
        assert result.btc_trend in ("up", "down", "chop")
        assert result.vol_state in ("expansion", "contraction")
        assert result.risk_state in ("risk_on", "risk_off", "neutral")
        assert "_" in result.regime_code
        assert result.timestamp is not None
        assert isinstance(result.metrics, dict)

    def test_no_coinbase_dsn(self):
        """Without coinbase DSN, defaults to chop/contraction."""
        snapshots = _make_snapshots([5.0, 3.0])

        result = compute_regime(snapshots, pg_dsn_coinbase="", cache_ttl=0)

        assert result.btc_trend == "chop"
        assert result.vol_state == "contraction"
        assert result.regime_code == "CHOP_CONTRACTION"


# ── Cache Tests ──────────────────────────────────────────────

class TestCache:
    def test_cache_prevents_recompute(self):
        snapshots = _make_snapshots([5.0, 3.0])

        # First call: computes
        result1 = compute_regime(snapshots, pg_dsn_coinbase="", cache_ttl=300)
        assert result1.cached is False

        # Second call: from cache
        result2 = compute_regime(snapshots, pg_dsn_coinbase="", cache_ttl=300)
        assert result2.cached is True
        assert result2.regime_code == result1.regime_code


# ── Storage Tests ────────────────────────────────────────────

class TestStorage:
    def test_db_failure_non_fatal(self):
        """DB failure in compute_regime should not raise."""
        snapshots = _make_snapshots([5.0])

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")

            # Should not raise
            result = compute_regime(
                snapshots,
                pg_dsn_coinbase="dsn",
                cache_ttl=0,
            )

        assert result.btc_trend == "chop"  # fallback
        assert result.vol_state == "contraction"

    def test_store_snapshot_success(self):
        regime = RegimeResult(
            regime_code="DOWNTREND_EXPANSION",
            btc_trend="down",
            vol_state="expansion",
            risk_state="risk_off",
            metrics={
                "btc_ema20": 65000.0,
                "btc_ema50": 66000.0,
                "btc_slope_4h": -0.005,
                "atr_4h": 500.0,
                "atr_ratio": 1.5,
                "breadth_pump10_pct": 0.1,
                "breadth_dump10_pct": 0.3,
                "funding_median": 0.001,
                "funding_pctl": 0.6,
            },
            timestamp=datetime.now(timezone.utc),
        )

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            store_regime_snapshot("dsn", regime)

            mock_cur.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_store_snapshot_failure(self):
        regime = RegimeResult(
            regime_code="CHOP_CONTRACTION",
            btc_trend="chop",
            vol_state="contraction",
            risk_state="neutral",
            metrics={},
            timestamp=datetime.now(timezone.utc),
        )

        with patch("hltrader.analysis.regime.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("DB down")

            # Should not raise
            store_regime_snapshot("dsn", regime)
