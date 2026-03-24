"""Trailing stop and break-even helpers."""

from __future__ import annotations

from typing import Optional

from hltrader.orders.validation import round_price
from hltrader.risk.stop_loss import get_position, move_stop


def set_break_even(coin: str, *, slippage: float = 0.05) -> dict:
    """Move the stop-loss to the entry price (break-even)."""
    pos = get_position(coin)
    if pos is None:
        raise ValueError(f"No open position for {coin}")
    return move_stop(coin, pos.entry_px, force=True, slippage=slippage)


def trailing_stop_step(
    coin: str,
    current_price: float,
    trail_pct: float,
    *,
    slippage: float = 0.05,
) -> Optional[dict]:
    """Ratchet the SL closer to price if the market has moved favourably.

    Returns the move_stop result if the SL was updated, else ``None``.
    """
    pos = get_position(coin)
    if pos is None:
        return None

    frac = trail_pct / 100.0
    if pos.is_long:
        new_sl = round_price(current_price * (1 - frac))
        # Only move up, never down
        if new_sl <= pos.entry_px:
            return None
    else:
        new_sl = round_price(current_price * (1 + frac))
        if new_sl >= pos.entry_px:
            return None

    # Check if the new SL is better than the current one
    from hltrader.risk.stop_loss import get_sl_orders_for_coin

    existing = get_sl_orders_for_coin(coin)
    if existing:
        current_sl = existing[0].trigger_px
        if pos.is_long and new_sl <= current_sl:
            return None  # new SL is worse (lower)
        if pos.is_short and new_sl >= current_sl:
            return None  # new SL is worse (higher)

    return move_stop(coin, new_sl, force=True, slippage=slippage)
