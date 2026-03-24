"""Tests for hltrader.scan.watcher cooldown logic."""

from __future__ import annotations

import time
from unittest.mock import patch

from hltrader.scan.watcher import (
    _last_alert_time,
    _last_extreme_alert_time,
    _record_alert,
    _record_extreme_alert,
    _should_alert,
    _should_alert_extreme,
)


class TestCooldownLogic:
    def setup_method(self):
        """Clear cooldown state before each test."""
        _last_alert_time.clear()

    def test_first_alert_always_allowed(self):
        assert _should_alert("BTC", cooldown=3600) is True

    def test_alert_blocked_within_cooldown(self):
        _record_alert("BTC")
        assert _should_alert("BTC", cooldown=3600) is False

    def test_alert_allowed_after_cooldown(self):
        _last_alert_time["BTC"] = time.time() - 4000  # 4000s ago
        assert _should_alert("BTC", cooldown=3600) is True

    def test_different_coins_independent(self):
        _record_alert("BTC")
        assert _should_alert("BTC", cooldown=3600) is False
        assert _should_alert("ETH", cooldown=3600) is True

    def test_record_updates_timestamp(self):
        _record_alert("BTC")
        t1 = _last_alert_time["BTC"]
        time.sleep(0.01)
        _record_alert("BTC")
        t2 = _last_alert_time["BTC"]
        assert t2 > t1

    def test_zero_cooldown_always_allows(self):
        _record_alert("BTC")
        assert _should_alert("BTC", cooldown=0) is True


class TestExtremeCooldownLogic:
    """Tests for the separate extreme lane cooldown."""

    def setup_method(self):
        """Clear both cooldown dicts before each test."""
        _last_alert_time.clear()
        _last_extreme_alert_time.clear()

    def test_first_extreme_alert_always_allowed(self):
        assert _should_alert_extreme("BTC", cooldown=21600) is True

    def test_extreme_blocked_within_cooldown(self):
        _record_extreme_alert("BTC")
        assert _should_alert_extreme("BTC", cooldown=21600) is False

    def test_extreme_allowed_after_cooldown(self):
        _last_extreme_alert_time["BTC"] = time.time() - 22000
        assert _should_alert_extreme("BTC", cooldown=21600) is True

    def test_extreme_and_normal_independent(self):
        """Normal and extreme cooldowns are completely independent."""
        _record_alert("BTC")
        # Normal cooldown active, but extreme should still be allowed
        assert _should_alert("BTC", cooldown=3600) is False
        assert _should_alert_extreme("BTC", cooldown=21600) is True

        # Now record extreme too
        _record_extreme_alert("BTC")
        assert _should_alert_extreme("BTC", cooldown=21600) is False

    def test_normal_unaffected_by_extreme(self):
        """Recording extreme alert doesn't block normal."""
        _record_extreme_alert("ETH")
        assert _should_alert("ETH", cooldown=3600) is True

    def test_extreme_different_coins_independent(self):
        _record_extreme_alert("BTC")
        assert _should_alert_extreme("BTC", cooldown=21600) is False
        assert _should_alert_extreme("SOL", cooldown=21600) is True

    def test_extreme_zero_cooldown_always_allows(self):
        _record_extreme_alert("BTC")
        assert _should_alert_extreme("BTC", cooldown=0) is True
