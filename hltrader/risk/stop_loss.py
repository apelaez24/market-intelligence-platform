"""Core stop-loss operations: query positions/orders, ensure/move/remove stops."""

from __future__ import annotations

from typing import Optional

from hltrader.client import get_exchange, get_info, get_address
from hltrader.models import CoinSpec, Position, TriggerOrderInfo
from hltrader.orders.trigger import place_stop_loss
from hltrader.orders.validation import validate_stop_loss


def get_all_positions() -> list[Position]:
    """Return all open positions for the configured account."""
    info = get_info()
    address = get_address()
    user_state = info.user_state(address)
    positions: list[Position] = []
    for item in user_state["assetPositions"]:
        if float(item["position"]["szi"]) != 0:
            positions.append(Position.from_user_state(item))
    return positions


def get_position(coin: str) -> Optional[Position]:
    """Return the position for *coin*, or ``None`` if flat."""
    for pos in get_all_positions():
        if pos.coin == coin:
            return pos
    return None


def get_coin_spec(coin: str) -> CoinSpec:
    """Fetch the CoinSpec (szDecimals, etc.) for *coin*."""
    info = get_info()
    meta = info.meta()
    for item in meta["universe"]:
        if item["name"] == coin:
            return CoinSpec.from_meta(item)
    raise ValueError(f"Coin {coin!r} not found")


def get_all_trigger_orders() -> list[TriggerOrderInfo]:
    """Return all open trigger (SL/TP) orders for the configured account."""
    info = get_info()
    address = get_address()
    orders = info.frontend_open_orders(address)
    triggers: list[TriggerOrderInfo] = []
    for o in orders:
        if o.get("isTrigger", False):
            triggers.append(TriggerOrderInfo.from_frontend_order(o))
    return triggers


def get_trigger_orders_for_coin(coin: str) -> list[TriggerOrderInfo]:
    """Return trigger orders that match *coin*."""
    return [t for t in get_all_trigger_orders() if t.coin == coin]


def get_sl_orders_for_coin(coin: str) -> list[TriggerOrderInfo]:
    """Return only SL-type trigger orders for *coin*."""
    return [
        t for t in get_trigger_orders_for_coin(coin)
        if t.reduce_only and ("sl" in t.order_type.lower() or "stop" in t.order_type.lower())
    ]


def ensure_stop_exists(
    coin: str,
    trigger_px: float,
    *,
    force: bool = False,
    slippage: float = 0.05,
) -> dict:
    """Place a stop-loss for *coin* if none exists yet.

    Returns ``{"action": "placed"|"exists", ...}``.
    """
    pos = get_position(coin)
    if pos is None:
        raise ValueError(f"No open position for {coin}")

    existing_sls = get_sl_orders_for_coin(coin)
    if existing_sls:
        return {"action": "exists", "trigger_orders": existing_sls}

    validate_stop_loss(pos.entry_px, trigger_px, pos.is_long, force=force)
    result = place_stop_loss(coin, trigger_px, pos.abs_size, pos.is_long, slippage=slippage)
    return {"action": "placed", "result": result}


def move_stop(
    coin: str,
    new_trigger_px: float,
    *,
    force: bool = False,
    slippage: float = 0.05,
) -> dict:
    """Cancel existing SL orders for *coin* and place a new one at *new_trigger_px*."""
    pos = get_position(coin)
    if pos is None:
        raise ValueError(f"No open position for {coin}")

    # Cancel existing SL triggers
    exchange = get_exchange()
    existing_sls = get_sl_orders_for_coin(coin)
    for sl in existing_sls:
        exchange.cancel(coin, sl.oid)

    validate_stop_loss(pos.entry_px, new_trigger_px, pos.is_long, force=force)
    result = place_stop_loss(coin, new_trigger_px, pos.abs_size, pos.is_long, slippage=slippage)
    return {"cancelled": len(existing_sls), "result": result}


def remove_stop(coin: str) -> int:
    """Cancel all SL trigger orders for *coin*. Returns count cancelled."""
    exchange = get_exchange()
    existing_sls = get_sl_orders_for_coin(coin)
    for sl in existing_sls:
        exchange.cancel(coin, sl.oid)
    return len(existing_sls)
