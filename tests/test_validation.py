"""Tests for orders/validation.py — pure logic, no SDK calls."""

from __future__ import annotations

import pytest

from hltrader.orders.validation import (
    compute_slippage_price,
    round_price,
    sl_trigger_px_from_pct,
    tp_trigger_px_from_pct,
    validate_stop_loss,
)


class TestValidateStopLoss:
    def test_long_sl_below_entry_ok(self):
        # Should not raise
        validate_stop_loss(entry_px=50000, trigger_px=49000, is_long=True)

    def test_long_sl_above_entry_raises(self):
        with pytest.raises(ValueError, match="below entry"):
            validate_stop_loss(entry_px=50000, trigger_px=51000, is_long=True)

    def test_long_sl_above_entry_force(self):
        # Should not raise with force=True
        validate_stop_loss(entry_px=50000, trigger_px=51000, is_long=True, force=True)

    def test_short_sl_above_entry_ok(self):
        validate_stop_loss(entry_px=50000, trigger_px=51000, is_long=False)

    def test_short_sl_below_entry_raises(self):
        with pytest.raises(ValueError, match="above entry"):
            validate_stop_loss(entry_px=50000, trigger_px=49000, is_long=False)

    def test_short_sl_below_entry_force(self):
        validate_stop_loss(entry_px=50000, trigger_px=49000, is_long=False, force=True)

    def test_sl_at_entry_raises_for_long(self):
        with pytest.raises(ValueError):
            validate_stop_loss(entry_px=50000, trigger_px=50000, is_long=True)

    def test_sl_at_entry_raises_for_short(self):
        with pytest.raises(ValueError):
            validate_stop_loss(entry_px=50000, trigger_px=50000, is_long=False)


class TestSlippagePrice:
    def test_buy_slippage_increases_price(self):
        result = compute_slippage_price(50000, is_buy=True, slippage=0.05)
        assert result > 50000

    def test_sell_slippage_decreases_price(self):
        result = compute_slippage_price(50000, is_buy=False, slippage=0.05)
        assert result < 50000

    def test_zero_slippage(self):
        result = compute_slippage_price(50000, is_buy=True, slippage=0.0)
        assert result == 50000


class TestRoundPrice:
    def test_round_price_sig_figs(self):
        assert round_price(48123.456789) == 48123.0

    def test_round_price_small(self):
        assert round_price(1.23456) == 1.2346


class TestSlTriggerPxFromPct:
    def test_long_2pct(self):
        result = sl_trigger_px_from_pct(50000, 2.0, is_long=True)
        assert result == round_price(49000)

    def test_short_2pct(self):
        result = sl_trigger_px_from_pct(50000, 2.0, is_long=False)
        assert result == round_price(51000)


class TestTpTriggerPxFromPct:
    def test_long_4pct(self):
        result = tp_trigger_px_from_pct(50000, 4.0, is_long=True)
        assert result == round_price(52000)

    def test_short_4pct(self):
        result = tp_trigger_px_from_pct(50000, 4.0, is_long=False)
        assert result == round_price(48000)
