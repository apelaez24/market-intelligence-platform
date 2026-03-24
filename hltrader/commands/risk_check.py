"""CLI commands: hl risk check [--fix], hl risk monitor"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hltrader.config import settings
from hltrader.risk.monitor import monitor_loop
from hltrader.risk.reconcile import reconcile

app = typer.Typer(name="risk", help="Risk auditing and monitoring")
console = Console()


@app.command()
def check(
    fix: bool = typer.Option(False, "--fix", help="Auto-place missing stop-losses"),
    sl_pct: Optional[float] = typer.Option(None, "--sl-pct", help="SL percentage for auto-fix"),
) -> None:
    """Audit all open positions for missing stop-loss orders."""
    rows = reconcile(fix=fix, sl_pct=sl_pct)

    if not rows:
        console.print("[dim]No open positions[/dim]")
        return

    table = Table(title="Risk Reconciliation")
    table.add_column("Coin", style="cyan")
    table.add_column("Side")
    table.add_column("Size", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Has SL", justify="center")
    table.add_column("SL Trigger", justify="right")
    table.add_column("Status")

    for row in rows:
        pos = row.position
        sl_px_str = ""
        if row.sl_orders:
            sl_px_str = str(row.sl_orders[0].trigger_px)

        if row.has_sl:
            status = "[green]OK[/green]"
        elif row.fixed:
            status = "[yellow]FIXED[/yellow]"
        elif row.error:
            status = f"[red]ERR: {row.error}[/red]"
        else:
            status = "[red]MISSING[/red]"

        table.add_row(
            pos.coin,
            "LONG" if pos.is_long else "SHORT",
            str(pos.abs_size),
            str(pos.entry_px),
            "Y" if row.has_sl else "N",
            sl_px_str,
            status,
        )

    console.print(table)


@app.command()
def monitor(
    interval: int = typer.Option(None, "--interval", "-i", help="Polling interval in seconds"),
    trail: Optional[float] = typer.Option(None, "--trail", "-t", help="Trailing stop percentage"),
) -> None:
    """Start a fallback polling monitor with optional trailing stops."""
    monitor_loop(interval=interval, trail_pct=trail)
