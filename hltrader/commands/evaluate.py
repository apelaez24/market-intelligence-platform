"""CLI command: hltrader evaluate outcomes / reset-null-evals / backfill."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Evaluate alert outcomes")
console = Console()


@app.command("outcomes")
def evaluate_outcomes(
    batch_size: int = typer.Option(100, "--batch-size", "-n",
                                   help="Max rows to evaluate per run"),
) -> None:
    """Evaluate pending alert outcomes using coinbase_data + HL fallback."""
    from hltrader.config import settings
    from hltrader.eval.outcomes import evaluate_pending

    if not settings.HL_ALERT_TRACKING_ENABLED:
        console.print("[yellow]HL_ALERT_TRACKING_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Evaluating pending alerts[/bold] (batch_size={batch_size})")
    summary = evaluate_pending(
        settings.pg_dsn,
        settings.pg_dsn_coinbase,
        batch_size=batch_size,
    )

    console.print(f"  1h evaluated: {summary['evaluated_1h']}")
    console.print(f"  4h evaluated: {summary['evaluated_4h']}")
    console.print(f"  24h evaluated: {summary['evaluated_24h']}")
    console.print(f"  HL fallback used: {summary.get('hl_evaluated', 0)}")
    console.print(f"  skipped: {summary['skipped']}")
    total = sum(v for k, v in summary.items() if k != "hl_evaluated")
    if total == 0:
        console.print("[dim]Nothing to evaluate[/dim]")
    else:
        console.print(f"[green]Done — {total} row(s) processed[/green]")


@app.command("reset-null-evals")
def reset_null_evals_cmd() -> None:
    """Reset evaluated flags where return values are NULL.

    Repairs rows marked evaluated but with no actual return data
    (e.g., non-Coinbase symbols before HL fallback was available).
    """
    from hltrader.config import settings
    from hltrader.eval.outcomes import reset_null_evals

    console.print("[bold]Resetting evaluated flags for NULL-return rows...[/bold]")
    result = reset_null_evals(settings.pg_dsn)

    console.print(f"  1h flags reset: {result['reset_1h']}")
    console.print(f"  4h flags reset: {result['reset_4h']}")
    console.print(f"  24h flags reset: {result['reset_24h']}")
    total = sum(result.values())
    if total == 0:
        console.print("[dim]No rows needed resetting[/dim]")
    else:
        console.print(f"[green]Done — {total} flag(s) reset. "
                      f"Run 'hltrader evaluate backfill' to re-evaluate.[/green]")


@app.command("backfill")
def backfill_cmd(
    limit: int = typer.Option(50, "--limit", "-n",
                              help="Max rows to process per run"),
) -> None:
    """Re-evaluate alerts with NULL returns using HL candle fallback.

    Processes pending (un-evaluated) alerts in batches.
    Use after 'reset-null-evals' to backfill historical data.
    """
    from hltrader.config import settings
    from hltrader.eval.outcomes import evaluate_pending

    if not settings.HL_ALERT_TRACKING_ENABLED:
        console.print("[yellow]HL_ALERT_TRACKING_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Backfilling pending alerts[/bold] (limit={limit})")
    console.print("[dim]Using HL candleSnapshot API for non-Coinbase symbols[/dim]")

    summary = evaluate_pending(
        settings.pg_dsn,
        settings.pg_dsn_coinbase,
        batch_size=limit,
    )

    console.print(f"  1h evaluated: {summary['evaluated_1h']}")
    console.print(f"  4h evaluated: {summary['evaluated_4h']}")
    console.print(f"  24h evaluated: {summary['evaluated_24h']}")
    console.print(f"  HL fallback used: {summary.get('hl_evaluated', 0)}")

    total = summary["evaluated_1h"] + summary["evaluated_4h"] + summary["evaluated_24h"]
    if total == 0:
        console.print("[dim]Nothing to backfill (all alerts evaluated or too recent)[/dim]")
    else:
        console.print(f"[green]Done — {total} evaluation(s) completed[/green]")


@app.command("stats")
def show_stats() -> None:
    """Show 7-day alert performance summary."""
    from hltrader.config import settings
    from hltrader.eval.outcomes import get_weekly_stats

    if not settings.HL_ALERT_TRACKING_ENABLED:
        console.print("[yellow]HL_ALERT_TRACKING_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    stats = get_weekly_stats(settings.pg_dsn)

    if stats["total_alerts"] == 0:
        console.print("[dim]No alerts in the last 7 days[/dim]")
        return

    console.print(f"\n[bold]Alert Performance (7d) — {stats['total_alerts']} alerts[/bold]\n")

    for t in stats["by_type"]:
        atype = t["alert_type"]
        total = t["total"]

        def _wr(wins, evald):
            if not evald:
                return "n/a"
            return f"{wins / evald * 100:.0f}%"

        wr1 = _wr(t["wins_1h"], t["eval_1h"])
        wr4 = _wr(t["wins_4h"], t["eval_4h"])
        wr24 = _wr(t["wins_24h"], t["eval_24h"])
        avg1 = f"{t['avg_1h']:+.1f}%" if t["avg_1h"] is not None else "n/a"
        avg24 = f"{t['avg_24h']:+.1f}%" if t["avg_24h"] is not None else "n/a"

        console.print(f"[bold]{atype}[/bold]: {total} alerts")
        console.print(f"  Win rate: 1h={wr1} 4h={wr4} 24h={wr24}")
        console.print(f"  Avg return: 1h={avg1} 24h={avg24}")

    if stats["top_symbols"]:
        top = ", ".join(f"{s['symbol']} {s['avg_ret']:+.1f}%" for s in stats["top_symbols"])
        console.print(f"\n[green]Top (best shorts):[/green] {top}")

    if stats["worst_symbols"]:
        worst = ", ".join(f"{s['symbol']} {s['avg_ret']:+.1f}%" for s in stats["worst_symbols"])
        console.print(f"[red]Worst:[/red] {worst}")
