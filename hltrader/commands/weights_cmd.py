"""CLI commands: hltrader weights now | hltrader weights compute."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Adaptive signal weight tools")
console = Console()


@app.command("now")
def weights_now() -> None:
    """Display current adaptive weights and what the watcher would use."""
    from hltrader.config import settings

    # Always show static defaults
    console.print("[bold]Static .env weights[/bold]")
    console.print(f"  pump: {settings.HL_SCAN_W_PUMP:.1f}  funding: {settings.HL_SCAN_W_FUNDING:.1f}"
                  f"  oi: {settings.HL_SCAN_W_OI:.1f}  accel: {settings.HL_SCAN_W_ACCEL:.1f}")
    console.print()

    if not settings.HL_ADAPTIVE_WEIGHTS_ENABLED:
        console.print("[yellow]HL_ADAPTIVE_WEIGHTS_ENABLED=false — watcher uses STATIC[/yellow]")
        raise typer.Exit(0)

    from hltrader.analysis.adaptive_weights import get_weights_for_regime, GLOBAL_REGIME
    import psycopg2

    # Show GLOBAL weights
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        try:
            from hltrader.analysis.adaptive_weights import _load_latest_with_meta
            gw, gd, gn = _load_latest_with_meta(conn, GLOBAL_REGIME)
            if gw:
                console.print(f"[bold]GLOBAL weights[/bold] (asof {gd}, n={gn})")
                console.print(f"  pump: {gw['pump_weight']:.3f}  oi: {gw['oi_weight']:.3f}"
                              f"  funding: {gw['funding_weight']:.3f}  accel: {gw['accel_weight']:.3f}")
            else:
                console.print("[dim]GLOBAL: no weights computed yet[/dim]")
            console.print()

            # Show any regime-specific weights
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DISTINCT ON (regime_code)
                              regime_code, date, sample_size,
                              pump_weight, oi_weight, funding_weight, accel_weight
                       FROM adaptive_signal_weights
                       WHERE regime_code != %s
                       ORDER BY regime_code, date DESC""",
                    (GLOBAL_REGIME,),
                )
                regime_rows = cur.fetchall()
            if regime_rows:
                console.print("[bold]Regime-specific weights[/bold]")
                for rc, d, n, pw, ow, fw, aw in regime_rows:
                    console.print(f"  {rc} (asof {d}, n={n}): P={pw:.3f} O={ow:.3f} F={fw:.3f} A={aw:.3f}")
                console.print()
        finally:
            conn.close()
    except Exception as exc:
        console.print(f"[red]DB error: {exc}[/red]")

    # Show what watcher would pick now
    sel = get_weights_for_regime(
        settings.pg_dsn,
        static_weights={
            "w_pump": settings.HL_SCAN_W_PUMP,
            "w_oi": settings.HL_SCAN_W_OI,
            "w_funding": settings.HL_SCAN_W_FUNDING,
            "w_accel": settings.HL_SCAN_W_ACCEL,
        },
    )
    w = sel.weights
    console.print(f"[bold green]Watcher would use: {sel.source}[/bold green]"
                  f" (asof {sel.asof_date or 'n/a'}, n={sel.sample_size})")
    console.print(f"  P={w['w_pump']:.1f} / O={w['w_oi']:.1f} / F={w['w_funding']:.1f} / A={w['w_accel']:.1f}")


@app.command("compute")
def weights_compute() -> None:
    """Compute and store adaptive weights from recent outcomes."""
    from hltrader.config import settings
    from hltrader.analysis.adaptive_weights import compute_adaptive_weights

    console.print(f"[dim]Computing adaptive weights (lookback={settings.HL_ADAPTIVE_WEIGHTS_LOOKBACK_DAYS}d, "
                  f"min_global={settings.HL_ADAPTIVE_WEIGHTS_MIN_GLOBAL}, "
                  f"min_regime={settings.HL_ADAPTIVE_WEIGHTS_MIN_PER_REGIME}, "
                  f"max_delta={settings.HL_ADAPTIVE_WEIGHTS_MAX_DAILY_DELTA})...[/dim]")

    results = compute_adaptive_weights(
        settings.pg_dsn,
        lookback_days=settings.HL_ADAPTIVE_WEIGHTS_LOOKBACK_DAYS,
        ema_alpha=settings.HL_ADAPTIVE_WEIGHTS_EMA_ALPHA,
        min_global=settings.HL_ADAPTIVE_WEIGHTS_MIN_GLOBAL,
        min_per_regime=settings.HL_ADAPTIVE_WEIGHTS_MIN_PER_REGIME,
        max_daily_delta=settings.HL_ADAPTIVE_WEIGHTS_MAX_DAILY_DELTA,
    )

    if not results:
        console.print("[yellow]Not enough evaluated alerts to compute weights[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Computed Adaptive Weights")
    table.add_column("Regime", style="bold")
    table.add_column("Pump", justify="right")
    table.add_column("OI", justify="right")
    table.add_column("Funding", justify="right")
    table.add_column("Accel", justify="right")
    table.add_column("N", justify="right")

    for aw in results:
        table.add_row(
            aw.regime_code,
            f"{aw.pump_weight:.3f}",
            f"{aw.oi_weight:.3f}",
            f"{aw.funding_weight:.3f}",
            f"{aw.accel_weight:.3f}",
            str(aw.sample_size),
        )

    console.print(table)
