"""CLI command: hltrader conviction test <symbol>."""

from __future__ import annotations

from datetime import datetime, timezone

import typer
from rich.console import Console

app = typer.Typer(help="Conviction score tools")
console = Console()


@app.command("test")
def conviction_test(
    symbol: str = typer.Argument(..., help="Symbol to test (e.g. BTC)"),
) -> None:
    """Compute and display conviction breakdown for a symbol."""
    from hltrader.config import settings

    if not settings.HL_CONVICTION_ENABLED:
        console.print("[yellow]HL_CONVICTION_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    from hltrader.analysis.conviction import compute_conviction

    result = compute_conviction(
        symbol=symbol.upper(),
        alert_type="NORMAL",
        composite_score=35.0,   # typical score
        pump_score=0.7,
        funding_score=0.5,
        oi_score=0.4,
        accel_score=0.2,
        liquidity=0.5,
        alert_timestamp=datetime.now(timezone.utc),
        pg_dsn=settings.pg_dsn,
        pg_dsn_coinbase=settings.pg_dsn_coinbase,
        w_base=settings.HL_CONVICTION_W_BASE,
        w_history=settings.HL_CONVICTION_W_HISTORY,
        w_regime=settings.HL_CONVICTION_W_REGIME,
        w_liquidity=settings.HL_CONVICTION_W_LIQUIDITY,
        w_geo=settings.HL_CONVICTION_W_GEO,
        min_sample=settings.HL_CONVICTION_MIN_SAMPLE,
        cache_ttl=settings.HL_CONVICTION_CACHE_TTL,
        history_prior=settings.HL_CONVICTION_HISTORY_PRIOR,
        conviction_min=settings.HL_CONVICTION_MIN,
    )

    c = result["components"]
    console.print(f"\n[bold]Conviction Breakdown: {symbol.upper()}[/bold]\n")
    console.print(f"  Base (composite):  {c['base']}")
    console.print(f"  History (edge):    {c['history']}")
    regime_label = result["regime"]
    console.print(f"  Regime (BTC):      {c['regime']}  ({regime_label})")
    console.print(f"  Liquidity:         {c['liquidity']}")
    console.print(f"  Geo alignment:     {c['geo']}")
    console.print(f"\n  [bold]Conviction: {result['conviction']}/100[/bold]")
    tier = result["tier"]
    if tier:
        color = {"A": "green", "B": "yellow", "C": "dim"}.get(tier, "white")
        console.print(f"  [bold {color}]Tier: {tier}[/bold {color}]")
    else:
        console.print("  [dim]Below threshold (no tier)[/dim]")

    # Phase 4.1: Display reasons
    reasons = result.get("reasons", [])
    if reasons:
        label = "Why allowed" if result.get("allowed") else "Why blocked"
        console.print(f"  {label}: {', '.join(reasons)}")
    if settings.HL_CONVICTION_SHADOW:
        console.print("  [dim]Mode: SHADOW (log only, no blocking)[/dim]")

    threshold = settings.HL_CONVICTION_MIN
    if result["conviction"] >= threshold:
        console.print(f"\n  [green]PASS — would send alert (>= {threshold})[/green]")
    else:
        console.print(f"\n  [red]BLOCKED — below threshold ({result['conviction']} < {threshold})[/red]")
