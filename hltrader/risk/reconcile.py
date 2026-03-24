"""Reconcile positions against trigger orders — find and fix gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hltrader.config import settings
from hltrader.models import Position, TriggerOrderInfo
from hltrader.orders.trigger import place_stop_loss
from hltrader.orders.validation import sl_trigger_px_from_pct, validate_stop_loss
from hltrader.risk.stop_loss import get_all_positions, get_all_trigger_orders


@dataclass
class ReconcileRow:
    coin: str
    position: Position
    sl_orders: list[TriggerOrderInfo]
    has_sl: bool
    fixed: bool = False
    error: Optional[str] = None


def reconcile(
    *,
    fix: bool = False,
    sl_pct: Optional[float] = None,
    slippage: float | None = None,
) -> list[ReconcileRow]:
    """Compare all positions to their trigger orders.

    If *fix* is True, auto-place missing stop-losses using *sl_pct*
    (or the configured default).
    """
    if sl_pct is None:
        sl_pct = settings.HL_DEFAULT_SL_PCT
    if slippage is None:
        slippage = settings.HL_DEFAULT_SLIPPAGE

    positions = get_all_positions()
    triggers = get_all_trigger_orders()

    # Index trigger orders by coin
    trigger_map: dict[str, list[TriggerOrderInfo]] = {}
    for t in triggers:
        trigger_map.setdefault(t.coin, []).append(t)

    rows: list[ReconcileRow] = []
    for pos in positions:
        sl_orders = [
            t for t in trigger_map.get(pos.coin, [])
            if t.reduce_only and ("sl" in t.order_type.lower() or "stop" in t.order_type.lower())
        ]
        has_sl = len(sl_orders) > 0
        row = ReconcileRow(
            coin=pos.coin,
            position=pos,
            sl_orders=sl_orders,
            has_sl=has_sl,
        )

        if not has_sl and fix:
            try:
                trigger_px = sl_trigger_px_from_pct(pos.entry_px, sl_pct, pos.is_long)
                validate_stop_loss(pos.entry_px, trigger_px, pos.is_long)
                place_stop_loss(pos.coin, trigger_px, pos.abs_size, pos.is_long, slippage=slippage)
                row.fixed = True
                row.has_sl = True
            except Exception as exc:
                row.error = str(exc)

        rows.append(row)

    return rows
