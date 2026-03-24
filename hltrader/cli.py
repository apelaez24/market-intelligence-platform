"""Typer CLI — wires command groups together as `hl`."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from hltrader.commands.bracket import app as bracket_app
from hltrader.commands.positions import app as pos_app
from hltrader.commands.risk_check import app as risk_app
from hltrader.commands.scan import app as scan_app
from hltrader.commands.selftest import app as selftest_app
from hltrader.commands.evaluate import app as eval_app
from hltrader.commands.conviction_cmd import app as conviction_app
from hltrader.commands.regime_cmd import app as regime_app
from hltrader.commands.state_cmd import app as state_app
from hltrader.commands.weights_cmd import app as weights_app
from hltrader.commands.token_memory_cmd import app as token_memory_app
from hltrader.commands.stop import app as stop_app

console = Console()

app = typer.Typer(
    name="hl",
    help="Hyperliquid stop-loss & risk management CLI",
    no_args_is_help=True,
)


@app.callback()
def main(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate orders — log what WOULD happen without sending anything"
    ),
    testnet: bool = typer.Option(
        False, "--testnet", help="Use Hyperliquid testnet instead of mainnet"
    ),
) -> None:
    """Global options applied before any subcommand."""
    from hltrader.config import settings

    if dry_run:
        settings.HL_DRY_RUN = True
    if testnet:
        settings.HL_TESTNET = True

    # Show banner when either mode is active
    if settings.HL_DRY_RUN or settings.HL_TESTNET:
        parts = []
        if settings.HL_DRY_RUN:
            parts.append("[bold magenta]DRY-RUN[/bold magenta]")
        if settings.HL_TESTNET:
            parts.append("[bold yellow]TESTNET[/bold yellow]")
        console.print(f"  Mode: {' + '.join(parts)}\n")


app.add_typer(stop_app, name="stop")
app.add_typer(bracket_app, name="bracket")
app.add_typer(risk_app, name="risk")
app.add_typer(pos_app, name="pos")
app.add_typer(scan_app, name="scan")
app.add_typer(selftest_app, name="selftest")
app.add_typer(eval_app, name="evaluate")
app.add_typer(conviction_app, name="conviction")
app.add_typer(regime_app, name="regime")
app.add_typer(state_app, name="state")
app.add_typer(weights_app, name="weights")
app.add_typer(token_memory_app, name="token-memory")


if __name__ == "__main__":
    app()
