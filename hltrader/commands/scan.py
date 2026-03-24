"""Scan Hyperliquid for coins with large 24h price moves (short candidates)."""

from __future__ import annotations

from typing import Optional

import requests
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Scan for large movers on Hyperliquid")
console = Console()

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _fetch_all_mids() -> dict[str, float]:
    """Get current mid prices for all assets."""
    resp = requests.post(
        HL_INFO_URL,
        json={"type": "allMids"},
        timeout=10,
    )
    resp.raise_for_status()
    return {k: float(v) for k, v in resp.json().items()}


def _fetch_24h_snapshots() -> list[dict]:
    """Get 24h context for all assets (day open, volume, etc.)."""
    resp = requests.post(
        HL_INFO_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    # data[0] = meta (universe), data[1] = list of asset contexts
    meta = data[0]["universe"]
    ctxs = data[1]

    results = []
    for asset_meta, ctx in zip(meta, ctxs):
        name = asset_meta["name"]
        try:
            mark_px = float(ctx.get("markPx", 0))
            prev_day_px = float(ctx.get("prevDayPx", 0))
            day_ntl_vlm = float(ctx.get("dayNtlVlm", 0))
            funding = float(ctx.get("funding", 0))
            open_interest = float(ctx.get("openInterest", 0))

            if prev_day_px > 0:
                pct_change = ((mark_px - prev_day_px) / prev_day_px) * 100
            else:
                pct_change = 0.0

            results.append({
                "coin": name,
                "price": mark_px,
                "prev_day_price": prev_day_px,
                "pct_24h": pct_change,
                "volume_24h": day_ntl_vlm,
                "funding_rate": funding,
                "open_interest": open_interest,
            })
        except (ValueError, TypeError):
            continue

    return results


@app.command("movers")
def scan_movers(
    threshold: float = typer.Option(
        20.0, "--threshold", "-t",
        help="Minimum absolute 24h % change to show (default 20%)",
    ),
    direction: str = typer.Option(
        "up", "--direction", "-d",
        help="Filter direction: 'up' (gainers), 'down' (losers), 'both'",
    ),
    min_volume: float = typer.Option(
        100_000, "--min-volume", "-v",
        help="Minimum 24h USD volume to include (default $100K)",
    ),
    limit: int = typer.Option(
        25, "--limit", "-n",
        help="Max results to show (default 25)",
    ),
) -> None:
    """Scan all Hyperliquid perps for large 24h movers."""
    with console.status("Fetching Hyperliquid market data..."):
        assets = _fetch_24h_snapshots()

    # Filter
    filtered = []
    for a in assets:
        if a["volume_24h"] < min_volume:
            continue
        if direction == "up" and a["pct_24h"] < threshold:
            continue
        if direction == "down" and a["pct_24h"] > -threshold:
            continue
        if direction == "both" and abs(a["pct_24h"]) < threshold:
            continue
        filtered.append(a)

    # Sort by % change descending (biggest gainers first)
    filtered.sort(key=lambda x: x["pct_24h"], reverse=True)
    filtered = filtered[:limit]

    if not filtered:
        console.print(f"[yellow]No coins found with >{threshold}% move and >${min_volume:,.0f} volume[/yellow]")
        return

    # Display
    table = Table(title=f"Hyperliquid 24h Movers (>{threshold}%)")
    table.add_column("Coin", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("24h %", justify="right")
    table.add_column("24h Vol", justify="right")
    table.add_column("Funding", justify="right")
    table.add_column("OI", justify="right")

    for a in filtered:
        pct_color = "green" if a["pct_24h"] > 0 else "red"
        funding_color = "red" if a["funding_rate"] > 0.01 else "green" if a["funding_rate"] < -0.01 else "white"

        table.add_row(
            a["coin"],
            f"${a['price']:,.4f}" if a["price"] < 1 else f"${a['price']:,.2f}",
            f"[{pct_color}]{a['pct_24h']:+.2f}%[/{pct_color}]",
            f"${a['volume_24h']:,.0f}",
            f"[{funding_color}]{a['funding_rate']:.4%}[/{funding_color}]",
            f"${a['open_interest']:,.0f}",
        )

    console.print(table)
    console.print(f"\n[dim]{len(filtered)} coins shown. "
                  f"These are potential short candidates if overextended.[/dim]")


@app.command("score")
def scan_score(
    min_score: float = typer.Option(
        None, "--min-score", "-s",
        help="Minimum composite score (0-100) to show (default from config)",
    ),
    min_volume: float = typer.Option(
        None, "--min-volume", "-v",
        help="Minimum 24h USD volume (default from config)",
    ),
    min_pct: float = typer.Option(
        None, "--min-pct",
        help="Minimum 24h %% change to consider (default from config)",
    ),
    limit: int = typer.Option(
        25, "--limit", "-n",
        help="Max results to show (default 25)",
    ),
) -> None:
    """One-shot scoring of short candidates (pump + funding + OI)."""
    from hltrader.config import settings
    from hltrader.scan.scorer import score_short_candidates
    from hltrader.scan.watcher import _print_candidates_table

    if min_score is None:
        min_score = settings.HL_SCAN_MIN_SCORE
    if min_volume is None:
        min_volume = settings.HL_SCAN_MIN_VOLUME_USD_24H
    if min_pct is None:
        min_pct = settings.HL_SCAN_MIN_PCT_24H

    with console.status("Fetching and scoring..."):
        snapshots = _fetch_24h_snapshots()
        candidates = score_short_candidates(
            snapshots,
            min_score=min_score,
            min_volume=min_volume,
            min_pct_24h=min_pct,
            min_oi=settings.HL_SCAN_MIN_OI_USD,
            pump_cap=settings.HL_SCAN_PUMP_CAP_PCT,
            pump_exp=settings.HL_SCAN_PUMP_EXP,
            liquidity_percentile_keep=settings.HL_SCAN_LIQUIDITY_PERCENTILE_KEEP,
            liquidity_metric=settings.HL_SCAN_LIQUIDITY_METRIC,
        )

    if not candidates:
        console.print("[yellow]No short candidates found[/yellow]")
        return

    _print_candidates_table(candidates[:limit])
    console.print(f"\n[dim]{len(candidates)} candidate(s) scored[/dim]")


@app.command("watch")
def scan_watch(
    interval: int = typer.Option(
        None, "--interval", "-i",
        help="Scan interval in seconds (default from config: 180)",
    ),
    min_score: float = typer.Option(
        None, "--min-score", "-s",
        help="Minimum composite score (0-100) for scanning (default from config).",
    ),
    min_volume: float = typer.Option(
        None, "--min-volume", "-v",
        help="Minimum 24h USD volume (default from config)",
    ),
    cooldown: int = typer.Option(
        None, "--cooldown", "-c",
        help="Per-coin alert cooldown in seconds (default from config)",
    ),
) -> None:
    """Periodically scan and alert on short candidates."""
    from hltrader.config import settings
    from hltrader.scan.watcher import watch_loop

    if interval is None:
        interval = settings.HL_SCAN_INTERVAL_SECONDS
    if min_score is None:
        min_score = settings.HL_SCAN_MIN_SCORE
    if min_volume is None:
        min_volume = settings.HL_SCAN_MIN_VOLUME_USD_24H
    if cooldown is None:
        cooldown = settings.HL_SCAN_COOLDOWN_SECONDS

    watch_loop(
        interval=interval,
        min_score=min_score,
        min_volume=min_volume,
        min_pct_24h=settings.HL_SCAN_MIN_PCT_24H,
        min_oi=settings.HL_SCAN_MIN_OI_USD,
        pump_cap=settings.HL_SCAN_PUMP_CAP_PCT,
        pump_exp=settings.HL_SCAN_PUMP_EXP,
        liquidity_percentile_keep=settings.HL_SCAN_LIQUIDITY_PERCENTILE_KEEP,
        liquidity_metric=settings.HL_SCAN_LIQUIDITY_METRIC,
        cooldown=cooldown,
        extreme_enabled=settings.HL_EXTREME_ENABLED,
        extreme_pct_24h=settings.HL_EXTREME_PCT_24H,
        extreme_score_threshold=settings.HL_EXTREME_SCORE_THRESHOLD,
        extreme_max_items=settings.HL_EXTREME_MAX_ITEMS,
        extreme_cooldown=settings.HL_EXTREME_COOLDOWN_SECONDS,
    )
