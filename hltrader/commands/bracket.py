"""CLI command: hl bracket open"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from hltrader.config import settings
from hltrader.orders.bracket import execute_bracket

app = typer.Typer(name="bracket", help="Bracket orders (entry + SL + optional TP)")
console = Console()


@app.command()
def open(
    coin: str = typer.Argument(..., help="Coin symbol (e.g. ETH)"),
    direction: str = typer.Argument(..., help="'long' or 'short'"),
    size: float = typer.Option(..., "--size", "-s", help="Position size in coin units"),
    lev: int = typer.Option(1, "--lev", "-l", help="Leverage"),
    sl_pct: Optional[float] = typer.Option(None, "--sl-pct", help="Stop-loss as % from entry"),
    sl_px: Optional[float] = typer.Option(None, "--sl-px", help="Stop-loss absolute price"),
    tp_pct: Optional[float] = typer.Option(None, "--tp-pct", help="Take-profit as % from entry"),
    tp_px: Optional[float] = typer.Option(None, "--tp-px", help="Take-profit absolute price"),
) -> None:
    """Open a market position with automatic stop-loss (and optional take-profit)."""
    coin = coin.upper()
    direction = direction.lower()
    if direction not in ("long", "short"):
        console.print("[red]Direction must be 'long' or 'short'[/red]")
        raise typer.Exit(1)

    is_long = direction == "long"

    if settings.HL_NO_STOP_NO_TRADE and sl_pct is None and sl_px is None:
        console.print(
            "[red]NO_STOP_NO_TRADE is enabled.[/red] "
            "You must provide --sl-pct or --sl-px."
        )
        raise typer.Exit(1)

    result = execute_bracket(
        coin,
        is_long,
        size,
        lev,
        sl_pct=sl_pct,
        sl_px=sl_px,
        tp_pct=tp_pct,
        tp_px=tp_px,
    )

    if result["position"] is None:
        console.print("[red]Bracket failed — position not detected after entry[/red]")
        raise typer.Exit(1)

    console.print("[bold green]Bracket order complete[/bold green]")
