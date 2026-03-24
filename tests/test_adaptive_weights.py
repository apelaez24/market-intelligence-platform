"""Tests for Phase 6/6.1: Adaptive Signal Weights."""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from hltrader.analysis.adaptive_weights import (
    DEFAULT_WEIGHTS,
    GLOBAL_REGIME,
    WEIGHT_KEYS,
    AdaptiveWeights,
    WeightsSelection,
    _clamp_deltas,
    _ema_smooth,
    _mean,
    _normalize_weights,
    _pearson,
    _scale_to_100,
    compute_adaptive_weights,
    get_weights_for_regime,
    load_adaptive_weights,
)


# ── Stats Tests ──────────────────────────────────────────────

class TestPearson:
    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(xs, ys) - 1.0) < 1e-10

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson(xs, ys) - (-1.0)) < 1e-10

    def test_no_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert _pearson(xs, ys) == 0.0

    def test_too_few_samples(self):
        assert _pearson([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_empty(self):
        assert _pearson([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _pearson([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0

    def test_moderate_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ys = [1.1, 2.3, 2.8, 4.2, 5.1, 5.8, 7.2]
        r = _pearson(xs, ys)
        assert 0.9 < r < 1.0


class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert _mean([]) == 0.0


# ── Weight Normalization ─────────────────────────────────────

class TestNormalizeWeights:
    def test_basic_normalization(self):
        raw = {"pump_weight": 0.4, "oi_weight": 0.3, "funding_weight": 0.2, "accel_weight": 0.1}
        n = _normalize_weights(raw)
        assert abs(sum(n.values()) - 1.0) < 1e-10

    def test_clamps_negatives(self):
        raw = {"pump_weight": 0.5, "oi_weight": -0.2, "funding_weight": 0.3, "accel_weight": 0.1}
        n = _normalize_weights(raw)
        assert n["oi_weight"] == 0.0
        assert abs(sum(n.values()) - 1.0) < 1e-10

    def test_all_zero_returns_defaults(self):
        raw = {"pump_weight": 0.0, "oi_weight": 0.0, "funding_weight": 0.0, "accel_weight": 0.0}
        n = _normalize_weights(raw)
        assert n == DEFAULT_WEIGHTS

    def test_all_negative_returns_defaults(self):
        raw = {"pump_weight": -0.3, "oi_weight": -0.1, "funding_weight": -0.5, "accel_weight": -0.2}
        n = _normalize_weights(raw)
        assert n == DEFAULT_WEIGHTS

    def test_preserves_proportions(self):
        raw = {"pump_weight": 2.0, "oi_weight": 1.0, "funding_weight": 1.0, "accel_weight": 0.0}
        n = _normalize_weights(raw)
        assert abs(n["pump_weight"] - 0.5) < 1e-10
        assert abs(n["oi_weight"] - 0.25) < 1e-10


# ── EMA Smoothing ────────────────────────────────────────────

class TestEMASmooth:
    def test_alpha_zero_keeps_old(self):
        old = {"pump_weight": 0.5, "oi_weight": 0.2, "funding_weight": 0.2, "accel_weight": 0.1}
        new = {"pump_weight": 0.1, "oi_weight": 0.4, "funding_weight": 0.3, "accel_weight": 0.2}
        result = _ema_smooth(old, new, alpha=0.0)
        for k in WEIGHT_KEYS:
            assert abs(result[k] - old[k]) < 1e-10

    def test_alpha_one_uses_new(self):
        old = {"pump_weight": 0.5, "oi_weight": 0.2, "funding_weight": 0.2, "accel_weight": 0.1}
        new = {"pump_weight": 0.1, "oi_weight": 0.4, "funding_weight": 0.3, "accel_weight": 0.2}
        result = _ema_smooth(old, new, alpha=1.0)
        for k in WEIGHT_KEYS:
            assert abs(result[k] - new[k]) < 1e-10

    def test_default_alpha(self):
        old = {"pump_weight": 0.5, "oi_weight": 0.2, "funding_weight": 0.2, "accel_weight": 0.1}
        new = {"pump_weight": 0.1, "oi_weight": 0.4, "funding_weight": 0.3, "accel_weight": 0.2}
        result = _ema_smooth(old, new, alpha=0.2)
        assert abs(result["pump_weight"] - 0.42) < 1e-10

    def test_missing_old_key_uses_default(self):
        old = {}
        new = {"pump_weight": 0.5, "oi_weight": 0.2, "funding_weight": 0.2, "accel_weight": 0.1}
        result = _ema_smooth(old, new, alpha=0.2)
        assert abs(result["pump_weight"] - 0.42) < 1e-10


# ── Clamp Deltas (Phase 6.1) ────────────────────────────────

class TestClampDeltas:
    def test_no_change_within_delta(self):
        old = {"pump_weight": 0.40, "oi_weight": 0.20, "funding_weight": 0.30, "accel_weight": 0.10}
        new = {"pump_weight": 0.45, "oi_weight": 0.18, "funding_weight": 0.27, "accel_weight": 0.10}
        result = _clamp_deltas(old, new, max_delta=0.08)
        # All deltas <= 0.08, so no clamping
        for k in WEIGHT_KEYS:
            assert abs(result[k] - new[k]) < 1e-10

    def test_clamps_large_positive_delta(self):
        old = {"pump_weight": 0.25, "oi_weight": 0.25, "funding_weight": 0.25, "accel_weight": 0.25}
        new = {"pump_weight": 0.60, "oi_weight": 0.25, "funding_weight": 0.25, "accel_weight": 0.25}
        result = _clamp_deltas(old, new, max_delta=0.08)
        # pump delta = 0.35, clamped to 0.08 → 0.33
        assert abs(result["pump_weight"] - 0.33) < 1e-10

    def test_clamps_large_negative_delta(self):
        old = {"pump_weight": 0.50, "oi_weight": 0.20, "funding_weight": 0.20, "accel_weight": 0.10}
        new = {"pump_weight": 0.10, "oi_weight": 0.20, "funding_weight": 0.20, "accel_weight": 0.10}
        result = _clamp_deltas(old, new, max_delta=0.08)
        # pump delta = -0.40, clamped to -0.08 → 0.42
        assert abs(result["pump_weight"] - 0.42) < 1e-10

    def test_clamp_respects_max_delta(self):
        old = {"pump_weight": 0.25, "oi_weight": 0.25, "funding_weight": 0.25, "accel_weight": 0.25}
        new = {"pump_weight": 0.50, "oi_weight": 0.50, "funding_weight": 0.00, "accel_weight": 0.00}
        result = _clamp_deltas(old, new, max_delta=0.08)
        for k in WEIGHT_KEYS:
            assert abs(result[k] - old[k]) <= 0.08 + 1e-10

    def test_zero_max_delta_freezes(self):
        old = {"pump_weight": 0.40, "oi_weight": 0.20, "funding_weight": 0.30, "accel_weight": 0.10}
        new = {"pump_weight": 0.10, "oi_weight": 0.50, "funding_weight": 0.10, "accel_weight": 0.30}
        result = _clamp_deltas(old, new, max_delta=0.0)
        for k in WEIGHT_KEYS:
            assert abs(result[k] - old[k]) < 1e-10


# ── Scale to 100 ─────────────────────────────────────────────

class TestScaleTo100:
    def test_basic(self):
        w = {"pump_weight": 0.4, "oi_weight": 0.2, "funding_weight": 0.3, "accel_weight": 0.1}
        s = _scale_to_100(w)
        assert abs(s["w_pump"] - 40.0) < 1e-10
        assert abs(sum(s.values()) - 100.0) < 1e-10


# ── Source Selection (Phase 6.1) ─────────────────────────────

class TestGetWeightsForRegime:
    def _mock_conn_with_rows(self, regime_rows=None, global_rows=None):
        """Helper to create a mock connection returning specific data."""
        mock_pg = MagicMock()
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # fetchone side effects: first call = regime lookup, second = GLOBAL lookup
        side_effects = []
        if regime_rows is not None:
            side_effects.append(regime_rows)
        else:
            side_effects.append(None)
        if global_rows is not None:
            side_effects.append(global_rows)
        else:
            side_effects.append(None)
        mock_cur.fetchone.side_effect = side_effects
        return mock_pg

    def test_source_regime_when_regime_exists(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            # Regime-specific found
            mock_cur.fetchone.return_value = (0.35, 0.25, 0.25, 0.15, date(2026, 3, 5), 48)

            sel = get_weights_for_regime("dsn", regime_code="UPTREND_EXPANSION")
            assert sel.source == "REGIME"
            assert sel.regime_code == "UPTREND_EXPANSION"
            assert sel.sample_size == 48
            assert abs(sel.weights["w_pump"] - 35.0) < 1e-10

    def test_source_global_when_regime_missing(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            # Regime miss, GLOBAL hit
            mock_cur.fetchone.side_effect = [None, (0.30, 0.30, 0.20, 0.20, date(2026, 3, 5), 100)]

            sel = get_weights_for_regime("dsn", regime_code="CHOP_CONTRACTION")
            assert sel.source == "GLOBAL"
            assert sel.regime_code == GLOBAL_REGIME
            assert sel.sample_size == 100

    def test_source_static_when_nothing_exists(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchone.return_value = None

            sel = get_weights_for_regime("dsn")
            assert sel.source == "STATIC"
            assert sel.asof_date is None
            assert sel.sample_size == 0

    def test_source_static_on_db_error(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("connection refused")
            sel = get_weights_for_regime("dsn")
            assert sel.source == "STATIC"

    def test_custom_static_weights_passed_through(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_pg.connect.side_effect = Exception("fail")
            custom = {"w_pump": 50.0, "w_oi": 10.0, "w_funding": 30.0, "w_accel": 10.0}
            sel = get_weights_for_regime("dsn", static_weights=custom)
            assert sel.source == "STATIC"
            assert sel.weights == custom


# ── Compute Tests ────────────────────────────────────────────

class TestComputeAdaptiveWeights:
    def test_global_skipped_when_below_min_global(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.description = [("pump_score",), ("oi_score",), ("funding_score",),
                                     ("accel_score",), ("eval_24h_return",), ("regime_code",)]
            mock_cur.fetchall.return_value = [
                (0.5, 0.3, 0.2, 0.1, -2.0, None) for _ in range(10)
            ]
            results = compute_adaptive_weights("dsn", min_global=30, min_per_regime=20)
            assert results == []

    def test_regime_skipped_when_below_min_per_regime(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.description = [("pump_score",), ("oi_score",), ("funding_score",),
                                     ("accel_score",), ("eval_24h_return",), ("regime_code",)]
            # 35 rows total (passes min_global=30), but only 5 for a specific regime
            rows = [(0.1 + i * 0.02, 0.2, 0.3, 0.1, -(0.1 + i * 0.02) * 10, None) for i in range(30)]
            rows += [(0.3, 0.2, 0.3, 0.1, -3.0, "UPTREND_EXPANSION") for _ in range(5)]
            mock_cur.fetchall.return_value = rows
            mock_cur.fetchone.return_value = None  # no prior weights

            results = compute_adaptive_weights("dsn", min_global=30, min_per_regime=20)
            # Should have GLOBAL but not UPTREND_EXPANSION
            regimes = [r.regime_code for r in results]
            assert GLOBAL_REGIME in regimes
            assert "UPTREND_EXPANSION" not in regimes

    def test_global_computed_and_stored(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.description = [("pump_score",), ("oi_score",), ("funding_score",),
                                     ("accel_score",), ("eval_24h_return",), ("regime_code",)]
            rows = []
            for i in range(35):
                pump = 0.1 + i * 0.02
                rows.append((pump, 0.2, 0.3, 0.1, -(pump * 10 + 1), None))
            mock_cur.fetchall.return_value = rows
            mock_cur.fetchone.return_value = None

            results = compute_adaptive_weights("dsn", min_global=30, min_per_regime=20)
            assert len(results) >= 1
            global_result = results[0]
            assert global_result.regime_code == GLOBAL_REGIME
            assert global_result.sample_size == 35
            total = (global_result.pump_weight + global_result.oi_weight +
                     global_result.funding_weight + global_result.accel_weight)
            assert abs(total - 1.0) < 0.01

    def test_deterministic_same_input_same_output(self):
        """Same input rows should produce identical weights."""
        rows = []
        for i in range(30):
            pump = 0.1 + i * 0.02
            rows.append({"pump_score": pump, "oi_score": 0.2, "funding_score": 0.3,
                          "accel_score": 0.1, "eval_24h_return": -(pump * 10 + 1),
                          "regime_code": None})

        from hltrader.analysis.adaptive_weights import _normalize_weights, _pearson, SIGNAL_COLUMNS
        # Run correlation twice
        results = []
        for _ in range(2):
            returns = [r["eval_24h_return"] for r in rows]
            raw_corr = {}
            for sig_col, wt_key in zip(SIGNAL_COLUMNS, WEIGHT_KEYS):
                signals = [r[sig_col] for r in rows]
                neg_returns = [-ret for ret in returns]
                raw_corr[wt_key] = _pearson(signals, neg_returns)
            results.append(_normalize_weights(raw_corr))

        for k in WEIGHT_KEYS:
            assert abs(results[0][k] - results[1][k]) < 1e-10


# ── Load (backward compat) ───────────────────────────────────

class TestLoadAdaptiveWeights:
    def test_returns_none_on_empty_db(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cur.fetchone.return_value = None

            result = load_adaptive_weights("dsn")
            assert result is None

    def test_returns_scaled_weights(self):
        with patch("hltrader.analysis.adaptive_weights.psycopg2") as mock_pg:
            mock_conn = MagicMock()
            mock_pg.connect.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            # Regime miss, GLOBAL hit
            mock_cur.fetchone.side_effect = [None, (0.35, 0.25, 0.25, 0.15, date(2026, 3, 5), 50)]

            result = load_adaptive_weights("dsn", regime_code="UPTREND_EXPANSION")
            assert result is not None
            assert abs(result["w_pump"] - 35.0) < 1e-10
            assert abs(sum(result.values()) - 100.0) < 1e-10
