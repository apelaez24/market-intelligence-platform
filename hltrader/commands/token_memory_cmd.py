"""CLI command: hltrader token-memory compute / now / top."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Token personality memory")
console = Console()


@app.command("compute")
def compute_cmd() -> None:
    """Recompute token personality memory from alert_outcomes."""
    from hltrader.config import settings
    from hltrader.analysis.token_memory import compute_token_memory

    if not settings.HL_TOKEN_MEMORY_ENABLED:
        console.print("[yellow]HL_TOKEN_MEMORY_ENABLED=false — skipping[/yellow]")
        raise typer.Exit(0)

    console.print("[bold]Computing token personality memory...[/bold]")
    summary = compute_token_memory(
        settings.pg_dsn,
        lookback_days=settings.HL_TOKEN_MEMORY_LOOKBACK_DAYS,
        min_sample=settings.HL_TOKEN_MEMORY_MIN_SAMPLE,
        min_sample_regime=settings.HL_TOKEN_MEMORY_MIN_SAMPLE_REGIME,
    )

    console.print(f"  Rows used: {summary['total_rows']}")
    console.print(f"  Symbols computed: {summary['computed']}")
    console.print(f"  Symbols skipped (< min_sample): {summary['skipped']}")
    if summary["computed"] > 0:
        console.print(f"[green]Done — {summary['computed']} personality profiles updated[/green]")
    else:
        console.print("[dim]No symbols met the minimum sample threshold[/dim]")


@app.command("now")
def show_symbol(
    symbol: str = typer.Argument(..., help="Symbol to look up (e.g. MEME)"),
) -> None:
    """Show token personality for a specific symbol."""
    from hltrader.config import settings
    from hltrader.analysis.token_memory import get_personality, clear_cache

    clear_cache()
    tp = get_personality(settings.pg_dsn, symbol.upper())

    if tp is None:
        console.print(f"[yellow]No personality data for {symbol.upper()}[/yellow]")
        console.print("[dim]Run 'hltrader token-memory compute' first, "
                      "or symbol may have < min_sample alerts[/dim]")
        return

    console.print(f"\n[bold]{tp.symbol}[/bold] — {tp.personality_label} "
                  f"(confidence: {tp.confidence_score:.0f}/100)\n")

    table = Table(show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Sample size", str(tp.sample_size))
    table.add_row("  Clustered", str(tp.sample_size_clustered))
    table.add_row("  Unclustered", str(tp.sample_size_unclustered))
    table.add_row("", "")

    def _pct(v):
        return f"{v:.1%}" if v is not None else "n/a"

    def _ret(v):
        return f"{v:+.2f}%" if v is not None else "n/a"

    table.add_row("Win rate 1h", _pct(tp.win_1h))
    table.add_row("Win rate 4h", _pct(tp.win_4h))
    table.add_row("Win rate 24h", _pct(tp.win_24h))
    table.add_row("", "")
    table.add_row("Avg return 1h", _ret(tp.avg_ret_1h))
    table.add_row("Avg return 4h", _ret(tp.avg_ret_4h))
    table.add_row("Avg return 24h", _ret(tp.avg_ret_24h))
    table.add_row("Avg MFE 24h", _ret(tp.avg_mfe_24h))
    table.add_row("Avg MAE 24h", _ret(tp.avg_mae_24h))
    table.add_row("", "")
    table.add_row("Trend follow", f"{tp.trend_follow_score:.0f}/100")
    table.add_row("Mean reversion", f"{tp.mean_reversion_score:.0f}/100")
    table.add_row("Reversal speed", f"{tp.reversal_speed_score:.0f}/100")
    table.add_row("Cluster sensitivity", f"{tp.cluster_sensitivity:+.2f}")
    table.add_row("", "")
    table.add_row("Best regime", tp.best_regime or "n/a")
    table.add_row("Worst regime", tp.worst_regime or "n/a")

    console.print(table)


@app.command("top")
def show_top(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of symbols"),
) -> None:
    """Show top token personalities ranked by confidence."""
    from hltrader.config import settings
    from hltrader.analysis.token_memory import clear_cache, _load_all_personalities, _memory_cache

    clear_cache()
    _load_all_personalities(settings.pg_dsn)

    if not _memory_cache:
        console.print("[dim]No token personality data. Run 'hltrader token-memory compute' first.[/dim]")
        return

    # Sort by confidence descending
    ranked = sorted(_memory_cache.values(), key=lambda tp: tp.confidence_score, reverse=True)

    table = Table(title=f"Token Personalities (top {min(limit, len(ranked))})")
    table.add_column("Symbol", style="bold")
    table.add_column("Label")
    table.add_column("Conf")
    table.add_column("N")
    table.add_column("Win24h")
    table.add_column("Avg24h")
    table.add_column("Follow")
    table.add_column("Fade")
    table.add_column("Speed")
    table.add_column("Best Regime")

    for tp in ranked[:limit]:
        win24 = f"{tp.win_24h:.0%}" if tp.win_24h is not None else "n/a"
        avg24 = f"{tp.avg_ret_24h:+.1f}%" if tp.avg_ret_24h is not None else "n/a"
        table.add_row(
            tp.symbol,
            tp.personality_label,
            f"{tp.confidence_score:.0f}",
            str(tp.sample_size),
            win24,
            avg24,
            f"{tp.trend_follow_score:.0f}",
            f"{tp.mean_reversion_score:.0f}",
            f"{tp.reversal_speed_score:.0f}",
            tp.best_regime or "-",
        )

    console.print(table)
