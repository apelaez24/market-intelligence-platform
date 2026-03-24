"""CLI command: hltrader state build."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Market state snapshot tools")
console = Console()


@app.command("build")
def state_build() -> None:
    """Build market_state.json + .md snapshot."""
    from hltrader.config import settings
    from hltrader.analysis.state_builder import build_state, write_state

    console.print("[dim]Building market state snapshot...[/dim]")

    state = build_state(settings.pg_dsn)
    json_path, md_path = write_state(state)

    # Print summary
    regime = state.get("regime")
    regime_str = regime["regime_code"] if regime else "UNAVAILABLE"
    risk_str = regime["risk_state"] if regime else "unknown"

    alerts = state.get("alerts", {})
    counts = alerts.get("last_24h_counts", {})
    total_alerts = sum(counts.values())
    top_count = len(alerts.get("top_active", []))

    perf = state.get("performance", {})
    tier_count = sum(t.get("count", 0) for t in perf.get("tiers", {}).values())

    geo = state.get("geo", {})
    geo_count = len(geo.get("top", []))

    services = state.get("system", {}).get("services", {})
    active_count = sum(1 for v in services.values() if v == "active")

    console.print(f"\n[bold]Market State Snapshot[/bold]")
    console.print(f"  Regime: [bold]{regime_str}[/bold] | Risk: {risk_str}")
    console.print(f"  Alerts (24h): {total_alerts} | Top signals: {top_count}")
    console.print(f"  Performance (7d): {tier_count} evaluated alerts")
    console.print(f"  Geo events: {geo_count}")
    console.print(f"  Services: {active_count}/{len(services)} active")

    errors = state.get("errors", [])
    if errors:
        for e in errors:
            console.print(f"  [yellow]Warning: {e}[/yellow]")

    console.print(f"\n  JSON: {json_path}")
    console.print(f"  MD:   {md_path}")
