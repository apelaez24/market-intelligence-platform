"""Tests for risk/stop_loss.py — mocked SDK calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hltrader.models import Position, TriggerOrderInfo
from hltrader.risk.stop_loss import (
    get_all_positions,
    get_position,
    get_sl_orders_for_coin,
    ensure_stop_exists,
    move_stop,
    remove_stop,
)


@pytest.fixture
def patch_client():
    """Patch client functions to return mocks."""
    with (
        patch("hltrader.risk.stop_loss.get_info") as mock_info,
        patch("hltrader.risk.stop_loss.get_address", return_value="0xabc") as mock_addr,
        patch("hltrader.risk.stop_loss.get_exchange") as mock_exchange,
    ):
        yield mock_info, mock_addr, mock_exchange


class TestGetAllPositions:
    def test_returns_positions(self, patch_client, mock_user_state):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = mock_user_state

        positions = get_all_positions()
        assert len(positions) == 1
        assert positions[0].coin == "BTC"
        assert positions[0].size == 0.1
        assert positions[0].is_long is True

    def test_empty_when_no_positions(self, patch_client):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = {
            "marginSummary": {"accountValue": "10000"},
            "assetPositions": [],
        }
        assert get_all_positions() == []


class TestGetPosition:
    def test_found(self, patch_client, mock_user_state):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = mock_user_state

        pos = get_position("BTC")
        assert pos is not None
        assert pos.coin == "BTC"

    def test_not_found(self, patch_client, mock_user_state):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = mock_user_state

        assert get_position("DOGE") is None


class TestEnsureStopExists:
    @patch("hltrader.risk.stop_loss.get_sl_orders_for_coin")
    @patch("hltrader.risk.stop_loss.place_stop_loss")
    def test_places_when_none_exists(
        self, mock_place, mock_sl_orders, patch_client, mock_user_state
    ):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = mock_user_state
        mock_sl_orders.return_value = []
        mock_place.return_value = {"response": {"data": {"statuses": ["ok"]}}}

        result = ensure_stop_exists("BTC", 49000.0)
        assert result["action"] == "placed"
        mock_place.assert_called_once()

    @patch("hltrader.risk.stop_loss.get_sl_orders_for_coin")
    def test_skips_when_sl_exists(self, mock_sl_orders, patch_client, mock_user_state):
        mock_info, _, _ = patch_client
        mock_info.return_value.user_state.return_value = mock_user_state
        mock_sl_orders.return_value = [MagicMock()]

        result = ensure_stop_exists("BTC", 49000.0)
        assert result["action"] == "exists"


class TestRemoveStop:
    @patch("hltrader.risk.stop_loss.get_sl_orders_for_coin")
    def test_cancels_orders(self, mock_sl_orders, patch_client):
        _, _, mock_exchange = patch_client
        order = MagicMock()
        order.oid = 123
        mock_sl_orders.return_value = [order]

        count = remove_stop("BTC")
        assert count == 1
        mock_exchange.return_value.cancel.assert_called_once_with("BTC", 123)
