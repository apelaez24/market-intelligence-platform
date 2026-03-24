"""CLI commands: hl pos list, hl pos close"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hltrader.client import get_exchange, get_info, get_address
from hltrader.risk.stop_loss import get_all_positions, get_position

app = typer.Typer(name="pos", help="Position management")
console = Console()


@app.command("list")
def list_positions() -> None:
    """List all open positions."""
    positions = get_all_positions()

    if not positions:
        console.print("[dim]No open positions[/dim]")
        return

    # Account summary
    info = get_info()
    address = get_address()
    user_state = info.user_state(address)
    acct_value = user_state["marginSummary"]["accountValue"]
    console.print(f"Account value: [bold]{acct_value}[/bold]\n")

    table = Table(title="Open Positions")
    table.add_column("Coin", style="cyan")
    table.add_column("Side")
    table.add_column("Size", justify="right")
    table.add_column("Entry Px", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("uPnL", justify="right")
    table.add_column("ROE %", justify="right")
    table.add_column("Lev", justify="right")
    table.add_column("Liq Px", justify="right")

    for pos in positions:
        roe_pct = f"{pos.return_on_equity * 100:.2f}%"
        roe_style = "green" if pos.return_on_equity >= 0 else "red"
        pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"

        table.add_row(
            pos.coin,
            "[green]LONG[/green]" if pos.is_long else "[red]SHORT[/red]",
            str(pos.abs_size),
            str(pos.entry_px),
            f"{pos.position_value:.2f}",
            f"[{pnl_style}]{pos.unrealized_pnl:.2f}[/{pnl_style}]",
            f"[{roe_style}]{roe_pct}[/{roe_style}]",
            f"{pos.leverage_value}x",
            str(pos.liquidation_px) if pos.liquidation_px else "-",
        )

    console.print(table)


@app.command()
def close(
    coin: str = typer.Argument(..., help="Coin symbol to close"),
    size: Optional[float] = typer.Option(None, "--size", "-s", help="Partial close size (default: full)"),
) -> None:
    """Market-close a position."""
    coin = coin.upper()
    pos = get_position(coin)
    if pos is None:
        console.print(f"[red]No open position for {coin}[/red]")
        raise typer.Exit(1)

    exchange = get_exchange()
    console.print(f"Closing {'partial ' if size else ''}{coin} position...")
    result = exchange.market_close(coin, sz=size)

    if result is None:
        console.print(f"[red]Position for {coin} not found by SDK[/red]")
        raise typer.Exit(1)

    try:
        status = result["response"]["data"]["statuses"][0]
        console.print(f"[green]Closed:[/green] {status}")
    except (KeyError, IndexError):
        console.print(f"[yellow]Result:[/yellow] {result}")
