"""Fallback polling monitor — checks prices against stops, trailing ratchet."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console

from hltrader.client import get_exchange, get_info
from hltrader.config import settings
from hltrader.risk.stop_loss import get_all_positions, get_sl_orders_for_coin
from hltrader.risk.trailing import trailing_stop_step

console = Console()

# Per-coin cooldown to avoid spamming close attempts
_last_close_attempt: dict[str, float] = {}
COOLDOWN_SECONDS = 30.0


def _should_close(coin: str) -> bool:
    last = _last_close_attempt.get(coin, 0.0)
    return (time.time() - last) >= COOLDOWN_SECONDS


def _record_close_attempt(coin: str) -> None:
    _last_close_attempt[coin] = time.time()


def monitor_loop(
    *,
    interval: Optional[int] = None,
    trail_pct: Optional[float] = None,
) -> None:
    """Blocking polling loop — runs until Ctrl-C.

    1. Fetch ``all_mids()`` for current prices.
    2. For each open position, check if the trigger SL has been breached
       (fallback in case the exchange trigger didn't fire).
    3. If *trail_pct* is set, ratchet trailing stops.
    """
    if interval is None:
        interval = settings.HL_MONITOR_INTERVAL_SECONDS
    info = get_info()
    exchange = get_exchange()

    console.print(f"[bold]Monitor started[/bold] — interval={interval}s, trail={trail_pct}%")
    console.print("[dim]Press Ctrl-C to stop[/dim]\n")

    try:
        while True:
            try:
                mids = info.all_mids()
                positions = get_all_positions()

                for pos in positions:
                    mid_str = mids.get(pos.coin)
                    if mid_str is None:
                        continue
                    mid = float(mid_str)

                    # Check SL breach as fallback
                    sl_orders = get_sl_orders_for_coin(pos.coin)
                    for sl in sl_orders:
                        breached = False
                        if pos.is_long and mid <= sl.trigger_px:
                            breached = True
                        elif pos.is_short and mid >= sl.trigger_px:
                            breached = True

                        if breached and _should_close(pos.coin):
                            console.print(
                                f"[red]FALLBACK CLOSE[/red] {pos.coin}: "
                                f"mid={mid}, SL trigger={sl.trigger_px}"
                            )
                            try:
                                exchange.market_close(pos.coin)
                                _record_close_attempt(pos.coin)
                            except Exception as exc:
                                console.print(f"[red]Close failed:[/red] {exc}")
                                _record_close_attempt(pos.coin)

                    # Trailing stop ratchet
                    if trail_pct is not None and trail_pct > 0:
                        result = trailing_stop_step(pos.coin, mid, trail_pct)
                        if result is not None:
                            console.print(
                                f"[yellow]Trailing SL updated[/yellow] {pos.coin}"
                            )

            except Exception as exc:
                console.print(f"[red]Monitor error:[/red] {exc}")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")
