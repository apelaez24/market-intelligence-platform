"""Shared fixtures for hltrader tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_user_state():
    """A realistic user_state response with one BTC long position."""
    return {
        "marginSummary": {
            "accountValue": "10000.00",
            "totalMarginUsed": "500.00",
            "totalNtlPos": "5000.00",
            "totalRawUsd": "5000.00",
        },
        "assetPositions": [
            {
                "position": {
                    "coin": "BTC",
                    "szi": "0.1",
                    "entryPx": "50000.0",
                    "unrealizedPnl": "100.0",
                    "returnOnEquity": "0.02",
                    "leverage": {"type": "cross", "value": 10},
                    "liquidationPx": "45000.0",
                    "marginUsed": "500.0",
                    "positionValue": "5000.0",
                },
                "type": "oneWay",
            }
        ],
        "crossMarginSummary": {
            "accountValue": "10000.00",
            "totalMarginUsed": "500.00",
            "totalNtlPos": "5000.00",
            "totalRawUsd": "5000.00",
        },
        "withdrawable": "9500.00",
    }


@pytest.fixture
def mock_frontend_orders():
    """A list of frontend_open_orders including one SL trigger."""
    return [
        {
            "coin": "BTC",
            "oid": 12345,
            "side": "A",
            "sz": "0.1",
            "limitPx": "46550.0",
            "triggerPx": "49000.0",
            "triggerCondition": "price <= 49000",
            "isTrigger": True,
            "reduceOnly": True,
            "orderType": "Stop Loss",
            "isPositionTpsl": False,
            "origSz": "0.1",
            "tif": "Ioc",
            "timestamp": 1700000000000,
        }
    ]


@pytest.fixture
def mock_meta():
    """A minimal meta() response."""
    return {
        "universe": [
            {"name": "BTC", "szDecimals": 4, "maxLeverage": 50},
            {"name": "ETH", "szDecimals": 3, "maxLeverage": 50},
            {"name": "SOL", "szDecimals": 1, "maxLeverage": 20},
        ]
    }
