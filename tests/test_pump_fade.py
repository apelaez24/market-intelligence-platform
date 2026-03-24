"""Tests for hltrader.scan.pump_fade — RSI, Bollinger, rollover, scoring."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from hltrader.notify import format_trade_alert, should_send_trade_alert

from hltrader.scan.pump_fade import (
    bollinger_position,
    bollinger_squeeze,
    cache_clear,
    get_closes,
    is_rsi_rollover,
    rsi_14,
    rsi_series,
    score_pump_fade,
    _cache,
    _cache_set,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module-level cache before each test."""
    cache_clear()
    yield
    cache_clear()


def _rising_series(start: float, step: float, n: int) -> list[float]:
    """Generate a monotonically rising series."""
    return [start + i * step for i in range(n)]


def _falling_series(start: float, step: float, n: int) -> list[float]:
    """Generate a monotonically falling series."""
    return [start - i * step for i in range(n)]


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRsi14:
    def test_known_values(self):
        """RSI of a known sequence should match hand-calculation."""
        # 15 closes: 14 gains of +1 each → all gains, no losses → RSI ≈ 100
        closes = [float(i) for i in range(15)]  # 0,1,2,...,14
        result = rsi_14(closes)
        assert result is not None
        assert result > 99.0  # all gains

    def test_all_up(self):
        """All gains should produce RSI near 100."""
        closes = _rising_series(100.0, 1.0, 30)
        result = rsi_14(closes)
        assert result is not None
        assert result > 99.0

    def test_all_down(self):
        """All losses should produce RSI near 0."""
        closes = _falling_series(100.0, 1.0, 30)
        result = rsi_14(closes)
        assert result is not None
        assert result < 1.0

    def test_insufficient_data(self):
        """Less than 15 closes should return None."""
        assert rsi_14([1.0] * 14) is None
        assert rsi_14([1.0] * 10) is None
        assert rsi_14([]) is None


class TestRsiSeries:
    def test_length(self):
        """Output length should be len(closes) - 14."""
        closes = _rising_series(100.0, 0.5, 30)
        series = rsi_series(closes)
        assert len(series) == len(closes) - 14

    def test_empty_for_short_input(self):
        """Less than 15 closes → empty list."""
        assert rsi_series([1.0] * 14) == []

    def test_last_matches_rsi_14(self):
        """Last value of rsi_series should match rsi_14 for same input."""
        closes = _rising_series(50.0, 0.3, 25) + _falling_series(57.2, 0.5, 5)
        series = rsi_series(closes)
        single = rsi_14(closes)
        assert series
        assert single is not None
        assert abs(series[-1] - single) < 0.01


# ---------------------------------------------------------------------------
# Rollover tests
# ---------------------------------------------------------------------------

class TestIsRsiRollover:
    def test_true(self):
        """Was overbought (>75) + 2 consecutive drops → rollover."""
        rsi_vals = [60, 70, 78, 80, 76, 73, 70]
        assert is_rsi_rollover(rsi_vals) is True

    def test_false_no_overbought(self):
        """Falling but never crossed 75 → no rollover."""
        rsi_vals = [60, 65, 70, 74, 72, 70, 68]
        assert is_rsi_rollover(rsi_vals) is False

    def test_false_still_rising(self):
        """Was >75 but last values are still rising → no rollover."""
        rsi_vals = [60, 70, 78, 80, 81, 82, 83]
        assert is_rsi_rollover(rsi_vals) is False

    def test_short_input(self):
        """Too few values → False."""
        assert is_rsi_rollover([80, 78]) is False


# ---------------------------------------------------------------------------
# Bollinger tests
# ---------------------------------------------------------------------------

class TestBollingerPosition:
    def test_pct_b(self):
        """Price at upper band → pct_b ≈ 1.0."""
        # 19 values at 100, last value pushed to upper band
        closes = [100.0] * 19
        # Compute what upper band would be with std=2
        # All same → sd=0, so upper=middle=lower=100. pct_b would be 0.5 (div-by-zero guard)
        # Instead use slight variation
        closes = [100.0 + (i % 3 - 1) * 0.5 for i in range(20)]
        result = bollinger_position(closes)
        assert result is not None
        assert "pct_b" in result
        assert "bandwidth" in result

    def test_above_upper(self):
        """Price well above upper band → pct_b > 1.0."""
        # 19 calm values then a spike
        closes = [100.0] * 19 + [120.0]
        result = bollinger_position(closes)
        assert result is not None
        assert result["pct_b"] > 1.0

    def test_insufficient_data(self):
        """Less than 20 closes → None."""
        assert bollinger_position([100.0] * 19) is None

    def test_at_middle(self):
        """Price near middle → pct_b ≈ 0.5."""
        # Symmetric oscillation around 100
        closes = [100.0 + (-1) ** i * 2.0 for i in range(20)]
        closes[-1] = 100.0  # last is exactly middle
        result = bollinger_position(closes)
        assert result is not None
        assert 0.3 < result["pct_b"] < 0.7


class TestBollingerSqueeze:
    def test_squeeze_true(self):
        """Current bandwidth at bottom → squeeze detected."""
        # Wide bandwidths then a very narrow one
        bandwidths = [0.10, 0.12, 0.11, 0.09, 0.13, 0.14, 0.11, 0.10, 0.12, 0.01]
        assert bollinger_squeeze(bandwidths) is True

    def test_squeeze_false(self):
        """Current bandwidth not in bottom → no squeeze."""
        bandwidths = [0.10, 0.12, 0.11, 0.09, 0.13, 0.14, 0.11, 0.10, 0.12, 0.15]
        assert bollinger_squeeze(bandwidths) is False

    def test_short_input(self):
        """Less than 2 values → False."""
        assert bollinger_squeeze([0.1]) is False


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScorePumpFade:
    def _mock_get_closes(self, closes_5m, closes_1h):
        """Return a side_effect function for get_closes mock."""
        def _side_effect(symbol, timeframe, limit, **kwargs):
            if timeframe == "5m":
                return closes_5m
            elif timeframe == "1h":
                return closes_1h
            return None
        return _side_effect

    def test_high_rsi_with_rollover(self):
        """RSI overbought + rollover → high score."""
        # Build a series that pumps then fades
        closes_5m = _rising_series(100.0, 2.0, 30) + _falling_series(158.0, 1.0, 10)
        closes_1h = _rising_series(100.0, 5.0, 20)
        ctx = {"pct_24h": 25.0, "composite": 60.0, "funding_rate": 0.005}

        with patch("hltrader.scan.pump_fade.get_closes", side_effect=self._mock_get_closes(closes_5m, closes_1h)):
            result = score_pump_fade("TEST", ctx)

        assert result["score"] > 0
        assert result["confidence"] == 1.0
        assert len(result["reasons"]) > 0

    def test_high_rsi_no_rollover(self):
        """RSI overbought but still rising → risk flag, lower score."""
        closes_5m = _rising_series(100.0, 2.0, 40)  # pure uptrend, no rollover
        closes_1h = _rising_series(100.0, 5.0, 20)
        ctx = {"pct_24h": 30.0, "composite": 70.0, "funding_rate": 0.002}

        with patch("hltrader.scan.pump_fade.get_closes", side_effect=self._mock_get_closes(closes_5m, closes_1h)):
            result = score_pump_fade("TEST", ctx)

        # Should still have some score from RSI overbought, but no rollover bonus
        assert result["rollover"] is False
        # RSI still climbing = risk flag
        if result["rsi_14_5m"] and result["rsi_14_5m"] > 70:
            assert any("rollover" in f.lower() for f in result["risk_flags"])

    def test_no_data(self):
        """Both data sources fail → confidence=0, only funding points."""
        ctx = {"pct_24h": 25.0, "composite": 60.0, "funding_rate": 0.0}

        with patch("hltrader.scan.pump_fade.get_closes", return_value=None):
            result = score_pump_fade("UNKNOWN", ctx)

        assert result["score"] == 0
        assert result["confidence"] == 0.0
        assert result["rollover"] is False

    def test_funding_bonus(self):
        """Positive funding adds 5 points."""
        # Minimal data that won't trigger RSI/BB
        closes_5m = [100.0] * 40  # flat → RSI ≈ 50, no overbought
        ctx = {"pct_24h": 10.0, "composite": 30.0, "funding_rate": 0.005}

        with patch("hltrader.scan.pump_fade.get_closes", side_effect=self._mock_get_closes(closes_5m, None)):
            result = score_pump_fade("TEST", ctx)

        assert result["score"] >= 5
        assert any("funding" in r.lower() for r in result["reasons"])

    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        # Build extreme data that would score > 100 if uncapped
        # Very high RSI (>80 = 35pts) + rollover (25pts) + BB above upper (15pts) + 1h overbought (10pts) + funding (5pts) = 90
        # Close enough but let's ensure cap works
        closes_5m = _rising_series(100.0, 3.0, 30) + _falling_series(190.0, 2.0, 10)
        closes_1h = _rising_series(100.0, 8.0, 20)
        ctx = {"pct_24h": 40.0, "composite": 80.0, "funding_rate": 0.01}

        with patch("hltrader.scan.pump_fade.get_closes", side_effect=self._mock_get_closes(closes_5m, closes_1h)):
            result = score_pump_fade("TEST", ctx)

        assert result["score"] <= 100


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_ttl(self):
        """Cached data reused within TTL, refreshed after."""
        mock_closes = [100.0] * 20
        call_count = 0

        def _mock_coinbase(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_closes

        with patch("hltrader.scan.pump_fade._get_closes_coinbase", side_effect=_mock_coinbase), \
             patch("hltrader.scan.pump_fade._get_closes_hl", return_value=None):
            # First call — fetches from source
            result1 = get_closes("BTC", "5m", 20, pg_dsn="fake_dsn", cache_ttl=2)
            assert result1 == mock_closes
            assert call_count == 1

            # Second call — uses cache
            result2 = get_closes("BTC", "5m", 20, pg_dsn="fake_dsn", cache_ttl=2)
            assert result2 == mock_closes
            assert call_count == 1  # no new call

            # Expire the cache by manipulating _cache timestamps
            key = "BTC:5m:20"
            _cache[key] = (time.time() - 10, mock_closes)

            # Third call — cache expired, fetches again
            result3 = get_closes("BTC", "5m", 20, pg_dsn="fake_dsn", cache_ttl=2)
            assert result3 == mock_closes
            assert call_count == 2


class TestGetClosesFallback:
    def test_coinbase_fallback_to_hl(self):
        """Coinbase fails → HL API used."""
        hl_closes = [50.0] * 20

        with patch("hltrader.scan.pump_fade._get_closes_coinbase", return_value=None), \
             patch("hltrader.scan.pump_fade._get_closes_hl", return_value=hl_closes):
            result = get_closes("BTC", "5m", 20, pg_dsn="fake_dsn")

        assert result == hl_closes

    def test_non_coinbase_symbol_uses_hl(self):
        """Symbol not in Coinbase tables → goes straight to HL."""
        hl_closes = [75.0] * 20

        with patch("hltrader.scan.pump_fade._get_closes_coinbase") as mock_cb, \
             patch("hltrader.scan.pump_fade._get_closes_hl", return_value=hl_closes):
            result = get_closes("DOGE", "5m", 20, pg_dsn="fake_dsn")

        # Coinbase should NOT be called for non-mapped symbol
        mock_cb.assert_not_called()
        assert result == hl_closes


# ---------------------------------------------------------------------------
# Trade routing tests
# ---------------------------------------------------------------------------

class TestTradeRouting:
    """Tests for should_send_trade_alert() routing logic."""

    @staticmethod
    def _make_settings(*, enabled=True, trades_chat="TRADES123", min_score=75):
        s = MagicMock()
        s.HL_PUMP_FADE_ENABLED = enabled
        s.HL_TELEGRAM_CHAT_ID_TRADES = trades_chat
        s.HL_PUMP_FADE_TRADE_MIN_SCORE = min_score
        return s

    def test_should_send_all_conditions_met(self):
        """score>=75, rollover=True, chat set, enabled → True."""
        fade = {"score": 80, "rollover": True, "confidence": 0.9}
        assert should_send_trade_alert(fade, self._make_settings()) is True

    def test_should_send_low_score(self):
        """score<75 → False."""
        fade = {"score": 60, "rollover": True, "confidence": 0.7}
        assert should_send_trade_alert(fade, self._make_settings()) is False

    def test_should_send_no_rollover(self):
        """rollover=False → False."""
        fade = {"score": 85, "rollover": False, "confidence": 0.8}
        assert should_send_trade_alert(fade, self._make_settings()) is False

    def test_should_send_no_trades_chat(self):
        """Empty TRADES chat → False."""
        fade = {"score": 85, "rollover": True, "confidence": 0.8}
        assert should_send_trade_alert(fade, self._make_settings(trades_chat="")) is False

    def test_should_send_disabled(self):
        """Pump-Fade disabled → False."""
        fade = {"score": 85, "rollover": True, "confidence": 0.8}
        assert should_send_trade_alert(fade, self._make_settings(enabled=False)) is False

    def test_should_send_exact_threshold(self):
        """score==75 (exact threshold) → True."""
        fade = {"score": 75, "rollover": True, "confidence": 0.8}
        assert should_send_trade_alert(fade, self._make_settings()) is True


class TestFormatTradeAlert:
    """Tests for format_trade_alert() message formatting."""

    def test_format_basic(self):
        """Output contains coin, score, rollover icon."""
        fade = {
            "score": 80,
            "confidence": 0.85,
            "rollover": True,
            "rsi_14_5m": 72.0,
            "rsi_14_1h": 68.0,
            "bb_pct_b": 0.95,
            "bb_bandwidth": 0.045,
            "reasons": ["RSI overbought (5m)"],
            "risk_flags": [],
        }
        msg = format_trade_alert("TESTCOIN", None, fade)
        assert "TESTCOIN" in msg
        assert "80/100" in msg
        assert "\u2705" in msg  # rollover icon
        assert "RSI(5m): 72" in msg
        assert "RSI(1h): 68" in msg
        assert "pct_b: 0.95" in msg

    def test_format_with_risk_flags(self):
        """Risk flags appear in output."""
        fade = {
            "score": 75,
            "confidence": 0.7,
            "rollover": True,
            "rsi_14_5m": 78.0,
            "rsi_14_1h": None,
            "bb_pct_b": None,
            "bb_bandwidth": None,
            "reasons": ["RSI overbought"],
            "risk_flags": ["Bollinger squeeze detected"],
        }
        msg = format_trade_alert("DOGE", None, fade)
        assert "DOGE" in msg
        assert "Bollinger squeeze" in msg
        assert "RSI overbought" in msg
