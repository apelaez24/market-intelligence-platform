"""CLI commands: hl stop set/move/remove/show"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hltrader.config import settings
from hltrader.orders.validation import sl_trigger_px_from_pct, validate_stop_loss
from hltrader.risk.stop_loss import (
    get_all_trigger_orders,
    get_position,
    get_sl_orders_for_coin,
    move_stop,
    remove_stop,
)
from hltrader.orders.trigger import place_stop_loss

app = typer.Typer(name="stop", help="Stop-loss management")
console = Console()


@app.command()
def set(
    coin: str = typer.Argument(..., help="Coin symbol (e.g. BTC, ETH)"),
    price: Optional[float] = typer.Option(None, "--price", "-p", help="Absolute SL trigger price"),
    pct: Optional[float] = typer.Option(None, "--pct", help="SL as percentage from entry (e.g. 2.5)"),
    force: bool = typer.Option(False, "--force", "-f", help="Override safety checks"),
    slippage: float = typer.Option(None, "--slippage", help="Slippage override"),
) -> None:
    """Place a native stop-loss trigger order."""
    coin = coin.upper()
    if slippage is None:
        slippage = settings.HL_DEFAULT_SLIPPAGE

    pos = get_position(coin)
    if pos is None:
        console.print(f"[red]No open position for {coin}[/red]")
        raise typer.Exit(1)

    if price is not None:
        trigger_px = price
    elif pct is not None:
        trigger_px = sl_trigger_px_from_pct(pos.entry_px, pct, pos.is_long)
    else:
        trigger_px = sl_trigger_px_from_pct(pos.entry_px, settings.HL_DEFAULT_SL_PCT, pos.is_long)

    try:
        validate_stop_loss(pos.entry_px, trigger_px, pos.is_long, force=force)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"Placing SL for {coin} ({'LONG' if pos.is_long else 'SHORT'}) "
        f"entry={pos.entry_px} trigger={trigger_px} size={pos.abs_size}"
    )

    result = place_stop_loss(coin, trigger_px, pos.abs_size, pos.is_long, slippage=slippage)
    try:
        status = result["response"]["data"]["statuses"][0]
        console.print(f"[green]SL placed:[/green] {status}")
    except (KeyError, IndexError):
        console.print(f"[yellow]Result:[/yellow] {result}")


@app.command()
def move(
    coin: str = typer.Argument(..., help="Coin symbol"),
    price: Optional[float] = typer.Option(None, "--price", "-p", help="New SL trigger price"),
    breakeven: bool = typer.Option(False, "--breakeven", "-b", help="Move SL to entry price"),
    slippage: float = typer.Option(None, "--slippage", help="Slippage override"),
) -> None:
    """Move an existing stop-loss to a new price."""
    coin = coin.upper()
    if slippage is None:
        slippage = settings.HL_DEFAULT_SLIPPAGE

    pos = get_position(coin)
    if pos is None:
        console.print(f"[red]No open position for {coin}[/red]")
        raise typer.Exit(1)

    if breakeven:
        new_px = pos.entry_px
    elif price is not None:
        new_px = price
    else:
        console.print("[red]Provide --price or --breakeven[/red]")
        raise typer.Exit(1)

    console.print(f"Moving SL for {coin} → {new_px}")
    result = move_stop(coin, new_px, force=breakeven, slippage=slippage)
    console.print(f"[green]Cancelled {result['cancelled']} old SL(s), new SL placed[/green]")


@app.command()
def remove(
    coin: str = typer.Argument(..., help="Coin symbol"),
) -> None:
    """Cancel all stop-loss orders for a coin."""
    coin = coin.upper()
    count = remove_stop(coin)
    if count:
        console.print(f"[green]Removed {count} SL order(s) for {coin}[/green]")
    else:
        console.print(f"[dim]No SL orders found for {coin}[/dim]")


@app.command()
def show(
    coin: Optional[str] = typer.Argument(None, help="Coin symbol (omit for all)"),
) -> None:
    """Show current stop-loss trigger orders."""
    if coin:
        coin = coin.upper()
        orders = get_sl_orders_for_coin(coin)
    else:
        orders = [t for t in get_all_trigger_orders() if t.reduce_only]

    if not orders:
        console.print("[dim]No SL trigger orders found[/dim]")
        return

    table = Table(title="Stop-Loss Trigger Orders")
    table.add_column("Coin", style="cyan")
    table.add_column("Side")
    table.add_column("Size", justify="right")
    table.add_column("Trigger Px", justify="right", style="red")
    table.add_column("Limit Px", justify="right")
    table.add_column("Type")
    table.add_column("OID", style="dim")

    for o in orders:
        table.add_row(
            o.coin,
            "SELL" if o.side == "A" else "BUY",
            str(o.size),
            str(o.trigger_px),
            str(o.limit_px),
            o.order_type,
            str(o.oid),
        )

    console.print(table)
