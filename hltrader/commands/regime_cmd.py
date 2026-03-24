"""CLI command: hltrader regime now."""

from __future__ import annotations

from datetime import datetime, timezone

import typer
from rich.console import Console

app = typer.Typer(help="Market regime tools")
console = Console()


@app.command("now")
def regime_now() -> None:
    """Compute and display current market regime."""
    from hltrader.config import settings

    if not settings.HL_REGIME_ENABLED:
        console.print("[yellow]HL_REGIME_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    from hltrader.analysis.regime import compute_regime, store_regime_snapshot
    from hltrader.commands.scan import _fetch_24h_snapshots

    console.print("[dim]Fetching snapshots...[/dim]")
    snapshots = _fetch_24h_snapshots()
    console.print(f"[dim]Got {len(snapshots)} assets[/dim]")

    result = compute_regime(
        snapshots,
        pg_dsn_coinbase=settings.pg_dsn_coinbase,
        slope_up=settings.HL_REGIME_SLOPE_UP,
        slope_down=settings.HL_REGIME_SLOPE_DOWN,
        atr_expand=settings.HL_REGIME_ATR_EXPAND,
        atr_contract=settings.HL_REGIME_ATR_CONTRACT,
        breadth_pump_threshold=settings.HL_REGIME_BREADTH_PUMP,
        breadth_dump_threshold=settings.HL_REGIME_BREADTH_DUMP,
        cache_ttl=settings.HL_REGIME_CACHE_TTL,
    )

    m = result.metrics
    console.print(f"\n[bold]Market Regime[/bold]\n")
    console.print(f"  Regime: [bold]{result.regime_code}[/bold]")
    console.print(f"  BTC Trend: {result.btc_trend}")
    console.print(f"  Vol State: {result.vol_state}")
    console.print(f"  Risk State: {result.risk_state}")

    console.print(f"\n[bold]Metrics[/bold]")
    if "btc_ema20" in m:
        console.print(f"  EMA20: ${m['btc_ema20']:,.2f}")
    if "btc_ema50" in m:
        console.print(f"  EMA50: ${m['btc_ema50']:,.2f}")
    if "btc_slope_4h" in m:
        console.print(f"  Slope (4h): {m['btc_slope_4h']:.6f}")
    if "atr_4h" in m:
        console.print(f"  ATR (4h): ${m['atr_4h']:,.2f}")
    if "atr_ratio" in m:
        console.print(f"  ATR Ratio: {m['atr_ratio']:.3f}")
    console.print(f"  Breadth pump>=10%: {m.get('breadth_pump10_pct', 0):.1%}")
    console.print(f"  Breadth dump<=-10%: {m.get('breadth_dump10_pct', 0):.1%}")
    console.print(f"  Funding median: {m.get('funding_median', 0):.6f}")
    console.print(f"  Funding +ve %: {m.get('funding_pctl', 0):.1%}")

    if settings.HL_REGIME_STORE_ENABLED:
        store_regime_snapshot(settings.pg_dsn, result)
        console.print(f"\n[dim]Snapshot stored to DB[/dim]")
