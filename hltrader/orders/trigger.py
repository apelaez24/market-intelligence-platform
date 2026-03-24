"""Place native server-side stop-loss and take-profit trigger orders."""

from __future__ import annotations

from typing import Any

from hltrader.client import get_exchange
from hltrader.orders.validation import compute_slippage_price, round_size


def place_stop_loss(
    coin: str,
    trigger_px: float,
    size: float,
    is_long: bool,
    *,
    slippage: float = 0.05,
) -> Any:
    """Place a native stop-loss trigger order.

    - ``is_long=True``  → sell to close → ``is_buy=False``
    - ``is_long=False`` → buy to close  → ``is_buy=True``
    - Always ``reduce_only=True``
    """
    exchange = get_exchange()
    is_buy = not is_long  # closing direction
    size = round_size(abs(size), coin)

    # Limit price with slippage so the IOC fill actually executes
    limit_px = compute_slippage_price(trigger_px, is_buy, slippage)

    order_type = {
        "trigger": {
            "triggerPx": trigger_px,
            "isMarket": True,
            "tpsl": "sl",
        }
    }

    result = exchange.order(
        coin,
        is_buy,
        size,
        limit_px,
        order_type,
        reduce_only=True,
    )
    return result


def place_take_profit(
    coin: str,
    trigger_px: float,
    size: float,
    is_long: bool,
    *,
    slippage: float = 0.05,
) -> Any:
    """Place a native take-profit trigger order.

    - ``is_long=True``  → sell to close → ``is_buy=False``
    - ``is_long=False`` → buy to close  → ``is_buy=True``
    - Always ``reduce_only=True``
    """
    exchange = get_exchange()
    is_buy = not is_long
    size = round_size(abs(size), coin)

    limit_px = compute_slippage_price(trigger_px, is_buy, slippage)

    order_type = {
        "trigger": {
            "triggerPx": trigger_px,
            "isMarket": True,
            "tpsl": "tp",
        }
    }

    result = exchange.order(
        coin,
        is_buy,
        size,
        limit_px,
        order_type,
        reduce_only=True,
    )
    return result
