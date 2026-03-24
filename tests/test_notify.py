"""Tests for hltrader.notify."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hltrader.notify import (
    format_extreme_pump_batch,
    format_short_alert_batch,
    is_telegram_configured,
    send_telegram,
)
from hltrader.scan.scorer import ShortCandidate


def _make_candidate(**kwargs):
    defaults = dict(
        coin="BTC",
        asset_type="Crypto",
        price=65000.0,
        pct_24h=25.0,
        funding_rate=0.005,
        open_interest=1_000_000.0,
        volume_24h=5_000_000.0,
        pump_score=0.625,
        funding_score=0.5,
        oi_score=0.8,
        composite=35.0,
    )
    defaults.update(kwargs)
    return ShortCandidate(**defaults)


class TestIsTelegramConfigured:
    def test_not_configured_when_empty(self):
        with patch("hltrader.notify.settings") as mock_settings:
            mock_settings.HL_TELEGRAM_BOT_TOKEN = ""
            mock_settings.HL_TELEGRAM_CHAT_ID = ""
            assert is_telegram_configured() is False

    def test_not_configured_when_partial(self):
        with patch("hltrader.notify.settings") as mock_settings:
            mock_settings.HL_TELEGRAM_BOT_TOKEN = "some-token"
            mock_settings.HL_TELEGRAM_CHAT_ID = ""
            assert is_telegram_configured() is False

    def test_configured_when_both_set(self):
        with patch("hltrader.notify.settings") as mock_settings:
            mock_settings.HL_TELEGRAM_BOT_TOKEN = "some-token"
            mock_settings.HL_TELEGRAM_CHAT_ID = "12345"
            assert is_telegram_configured() is True


class TestSendTelegram:
    @patch("hltrader.notify.requests.post")
    def test_sends_message(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 1}}
        mock_post.return_value = mock_resp

        with patch("hltrader.notify.settings") as mock_settings:
            mock_settings.HL_TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.HL_TELEGRAM_CHAT_ID = "12345"
            result = send_telegram("Hello!")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "test-token" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"]["text"] == "Hello!"
        assert call_kwargs.kwargs["json"]["parse_mode"] == "HTML"

    @patch("hltrader.notify.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_resp

        with patch("hltrader.notify.settings") as mock_settings:
            mock_settings.HL_TELEGRAM_BOT_TOKEN = "bad-token"
            mock_settings.HL_TELEGRAM_CHAT_ID = "12345"
            with pytest.raises(Exception, match="401"):
                send_telegram("Hello!")


class TestFormatShortAlertBatch:
    def test_single_candidate(self):
        msg = format_short_alert_batch([_make_candidate(composite=35.0)])
        assert "<b>Short Candidates Detected</b>" in msg
        assert "BTC" in msg
        assert "35" in msg  # composite score
        assert "+25.0%" in msg

    def test_multiple_candidates(self):
        candidates = [
            _make_candidate(coin="BTC", composite=40.0),
            _make_candidate(coin="ETH", composite=30.0, price=3500.0),
        ]
        msg = format_short_alert_batch(candidates)
        assert "BTC" in msg
        assert "ETH" in msg

    def test_empty_list(self):
        msg = format_short_alert_batch([])
        assert msg == ""

    def test_above_cap_filtered_in_notify(self):
        """Defense-in-depth: candidates > 45 are dropped even in notify."""
        candidates = [
            _make_candidate(coin="HIGH", composite=80.0),
            _make_candidate(coin="LOW", composite=30.0),
        ]
        msg = format_short_alert_batch(candidates)
        assert "LOW" in msg
        assert "HIGH" not in msg


class TestFormatExtremePumpBatch:
    """Tests for the extreme pump Telegram message formatter."""

    def test_single_extreme_candidate(self):
        msg = format_extreme_pump_batch([_make_candidate(composite=60.0, pct_24h=30.0)])
        assert "EXTREME PUMP WATCH" in msg
        assert "BTC" in msg
        assert "60" in msg  # score
        assert "+30.0%" in msg
        assert "reversal confirmation" in msg
        assert "Why flagged:" in msg
        assert "Playbook:" in msg

    def test_multiple_extreme_candidates(self):
        candidates = [
            _make_candidate(coin="DOGE", composite=70.0, pct_24h=35.0),
            _make_candidate(coin="SHIB", composite=55.0, pct_24h=22.0),
        ]
        msg = format_extreme_pump_batch(candidates)
        assert "DOGE" in msg
        assert "SHIB" in msg
        assert "EXTREME PUMP WATCH" in msg

    def test_empty_list_returns_empty(self):
        msg = format_extreme_pump_batch([])
        assert msg == ""

    def test_no_score_cap_filtering(self):
        """Extreme format does NOT filter by MAX_ALERT_SCORE — all pass through."""
        candidates = [
            _make_candidate(coin="MEGA", composite=95.0, pct_24h=50.0),
        ]
        msg = format_extreme_pump_batch(candidates)
        assert "MEGA" in msg
        assert "95" in msg

    def test_includes_risk_warning(self):
        """Each extreme candidate includes a risk/reversal warning line."""
        msg = format_extreme_pump_batch([_make_candidate(composite=60.0)])
        assert "Do NOT front-run" in msg
        assert "reversal confirmation.</i>" in msg

    def test_asset_type_emoji(self):
        """Correct emoji for different asset types."""
        crypto = format_extreme_pump_batch([_make_candidate(asset_type="Crypto", composite=60.0)])
        assert "\U0001f4b0" in crypto  # money bag

        commodity = format_extreme_pump_batch([_make_candidate(asset_type="Commodity", composite=60.0)])
        assert "\U0001f947" in commodity  # gold medal

    def test_previously_dropped_candidate_now_alerted(self):
        """Before/after: a candidate with score 60 was silently dropped by
        normal lane. Now it appears in EXTREME PUMP WATCH."""
        candidate = _make_candidate(coin="PUMP_COIN", composite=60.0, pct_24h=28.0)

        # Normal lane would drop it
        normal_msg = format_short_alert_batch([candidate])
        assert normal_msg == ""  # dropped by defense-in-depth

        # Extreme lane shows it
        extreme_msg = format_extreme_pump_batch([candidate])
        assert "PUMP_COIN" in extreme_msg
        assert "EXTREME PUMP WATCH" in extreme_msg
        assert "60" in extreme_msg
