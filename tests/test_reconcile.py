"""Tests for risk/reconcile.py — mocked SDK calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hltrader.models import Position, TriggerOrderInfo
from hltrader.risk.reconcile import reconcile


@pytest.fixture
def btc_position():
    return Position(
        coin="BTC",
        size=0.1,
        entry_px=50000.0,
        unrealized_pnl=100.0,
        return_on_equity=0.02,
        leverage_type="cross",
        leverage_value=10,
        liquidation_px=45000.0,
        margin_used=500.0,
        position_value=5000.0,
    )


@pytest.fixture
def btc_sl_trigger():
    return TriggerOrderInfo(
        oid=12345,
        coin="BTC",
        side="A",
        size=0.1,
        limit_px=46550.0,
        trigger_px=49000.0,
        trigger_condition="price <= 49000",
        is_trigger=True,
        reduce_only=True,
        order_type="Stop Loss",
        is_position_tpsl=False,
    )


class TestReconcile:
    @patch("hltrader.risk.reconcile.get_all_trigger_orders")
    @patch("hltrader.risk.reconcile.get_all_positions")
    def test_all_covered(self, mock_positions, mock_triggers, btc_position, btc_sl_trigger):
        mock_positions.return_value = [btc_position]
        mock_triggers.return_value = [btc_sl_trigger]

        rows = reconcile()
        assert len(rows) == 1
        assert rows[0].has_sl is True
        assert rows[0].fixed is False

    @patch("hltrader.risk.reconcile.get_all_trigger_orders")
    @patch("hltrader.risk.reconcile.get_all_positions")
    def test_missing_sl_no_fix(self, mock_positions, mock_triggers, btc_position):
        mock_positions.return_value = [btc_position]
        mock_triggers.return_value = []

        rows = reconcile(fix=False)
        assert len(rows) == 1
        assert rows[0].has_sl is False
        assert rows[0].fixed is False

    @patch("hltrader.risk.reconcile.place_stop_loss")
    @patch("hltrader.risk.reconcile.get_all_trigger_orders")
    @patch("hltrader.risk.reconcile.get_all_positions")
    def test_missing_sl_with_fix(self, mock_positions, mock_triggers, mock_place, btc_position):
        mock_positions.return_value = [btc_position]
        mock_triggers.return_value = []
        mock_place.return_value = {"response": {"data": {"statuses": ["ok"]}}}

        rows = reconcile(fix=True, sl_pct=2.0)
        assert len(rows) == 1
        assert rows[0].fixed is True
        mock_place.assert_called_once()

    @patch("hltrader.risk.reconcile.get_all_trigger_orders")
    @patch("hltrader.risk.reconcile.get_all_positions")
    def test_no_positions(self, mock_positions, mock_triggers):
        mock_positions.return_value = []
        mock_triggers.return_value = []

        rows = reconcile()
        assert rows == []
