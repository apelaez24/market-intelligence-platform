"""Bracket orders: market entry → poll for fill → attach SL + optional TP."""

from __future__ import annotations

import time
from typing import Any, Optional

from rich.console import Console

from hltrader.client import get_exchange, get_info, get_address
from hltrader.config import settings
from hltrader.models import Position
from hltrader.orders.trigger import place_stop_loss, place_take_profit
from hltrader.orders.validation import (
    round_size,
    sl_trigger_px_from_pct,
    tp_trigger_px_from_pct,
    validate_stop_loss,
)

console = Console()


def _wait_for_fill(coin: str, timeout: int = 30, poll_interval: float = 1.0) -> Optional[Position]:
    """Poll user_state until a position appears for *coin*, or timeout."""
    info = get_info()
    address = get_address()
    deadline = time.time() + timeout
    while time.time() < deadline:
        user_state = info.user_state(address)
        for item in user_state["assetPositions"]:
            pos = item["position"]
            if pos["coin"] == coin and float(pos["szi"]) != 0:
                return Position.from_user_state(item)
        time.sleep(poll_interval)
    return None


def execute_bracket(
    coin: str,
    is_long: bool,
    size: float,
    leverage: int,
    *,
    sl_pct: Optional[float] = None,
    sl_px: Optional[float] = None,
    tp_pct: Optional[float] = None,
    tp_px: Optional[float] = None,
    slippage: float | None = None,
) -> dict[str, Any]:
    """Open a market position and immediately attach SL (+ optional TP).

    Returns a dict with keys: ``entry_result``, ``position``, ``sl_result``, ``tp_result``.
    """
    if slippage is None:
        slippage = settings.HL_DEFAULT_SLIPPAGE

    exchange = get_exchange()
    size = round_size(size, coin)

    # 1. Set leverage
    exchange.update_leverage(leverage, coin)

    # 2. Market entry
    console.print(f"[bold]Opening market {'LONG' if is_long else 'SHORT'}[/bold] {size} {coin} @ {leverage}x")
    entry_result = exchange.market_open(coin, is_long, size, slippage=slippage)
    console.print(f"Entry result: {entry_result['response']['data']['statuses'][0]}")

    # 3. Wait for fill
    console.print("[dim]Waiting for fill...[/dim]")
    position = _wait_for_fill(coin)
    if position is None:
        console.print("[red]Timed out waiting for position fill[/red]")
        return {"entry_result": entry_result, "position": None, "sl_result": None, "tp_result": None}

    entry_px = position.entry_px
    pos_size = position.abs_size
    console.print(f"[green]Filled:[/green] {position.coin} size={position.size} entry={entry_px}")

    # 4. Compute SL trigger price
    if sl_px is not None:
        sl_trigger = sl_px
    elif sl_pct is not None:
        sl_trigger = sl_trigger_px_from_pct(entry_px, sl_pct, is_long)
    else:
        sl_trigger = sl_trigger_px_from_pct(entry_px, settings.HL_DEFAULT_SL_PCT, is_long)

    validate_stop_loss(entry_px, sl_trigger, is_long)
    console.print(f"[yellow]Placing SL[/yellow] @ {sl_trigger}")
    sl_result = place_stop_loss(coin, sl_trigger, pos_size, is_long, slippage=slippage)
    console.print(f"SL result: {sl_result['response']['data']['statuses'][0]}")

    # 5. Optional TP
    tp_result = None
    if tp_px is not None or tp_pct is not None:
        if tp_px is not None:
            tp_trigger = tp_px
        else:
            tp_trigger = tp_trigger_px_from_pct(entry_px, tp_pct, is_long)  # type: ignore[arg-type]
        console.print(f"[green]Placing TP[/green] @ {tp_trigger}")
        tp_result = place_take_profit(coin, tp_trigger, pos_size, is_long, slippage=slippage)
        console.print(f"TP result: {tp_result['response']['data']['statuses'][0]}")

    return {
        "entry_result": entry_result,
        "position": position,
        "sl_result": sl_result,
        "tp_result": tp_result,
    }
