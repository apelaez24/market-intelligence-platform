"""Periodic watcher — scans, scores, and alerts on short candidates."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table

from hltrader.commands.scan import _fetch_24h_snapshots
from hltrader.notify import (
    format_cluster_alert,
    format_extreme_pump_batch,
    format_short_alert_batch,
    format_trade_alert,
    is_telegram_configured,
    send_telegram,
    should_send_trade_alert,
)
from hltrader.scan.cluster import detect_clusters
from hltrader.scan.candles import build_returns_cache, set_cache_ttl
from hltrader.scan.scorer import (
    classify_extreme_candidates,
    filter_alert_candidates,
    score_short_candidates,
)

console = Console()
log = logging.getLogger(__name__)

# Per-coin cooldown to avoid re-alerting the same coin (normal lane)
_last_alert_time: dict[str, float] = {}

# Separate cooldown for extreme lane
_last_extreme_alert_time: dict[str, float] = {}


def _should_alert(coin: str, cooldown: float) -> bool:
    last = _last_alert_time.get(coin, 0.0)
    return (time.time() - last) >= cooldown


def _record_alert(coin: str) -> None:
    _last_alert_time[coin] = time.time()


def _should_alert_extreme(coin: str, cooldown: float) -> bool:
    last = _last_extreme_alert_time.get(coin, 0.0)
    return (time.time() - last) >= cooldown


def _record_extreme_alert(coin: str) -> None:
    _last_extreme_alert_time[coin] = time.time()


# Separate cooldown for TRADES chat
_last_trade_alert_time: dict[str, float] = {}


def _should_alert_trade(coin: str, cooldown: float) -> bool:
    last = _last_trade_alert_time.get(coin, 0.0)
    return (time.time() - last) >= cooldown


def _record_trade_alert(coin: str) -> None:
    _last_trade_alert_time[coin] = time.time()


def _record_xdedupe_keys(candidates: list) -> None:
    """Record cross-dedupe keys in DB for inter-service coordination.

    Non-fatal: if DB is unavailable the alert still went out.
    """
    from hltrader.config import settings

    if not settings.HL_ALERT_XDEDUPE_ENABLED:
        return
    try:
        import psycopg2
        from hltrader.alert_xdedupe import record_alert_key

        xconn = psycopg2.connect(settings.pg_dsn)
        xconn.autocommit = False
        try:
            for c in candidates:
                direction = "PUMP" if c.pct_24h >= 0 else "DROP"
                record_alert_key(
                    xconn,
                    symbol=c.coin,
                    direction=direction,
                    alert_type="EXTREME",
                    source="hltrader-watcher",
                )
        finally:
            xconn.close()
        console.print(f"[dim]xdedupe: recorded {len(candidates)} key(s)[/dim]")
    except Exception as exc:
        console.print(f"[dim]xdedupe record failed (non-fatal): {exc}[/dim]")


def _record_alert_outcomes(candidates: list, alert_type: str, *, cluster_type: str | None = None) -> None:
    """Record alert outcomes for tracking. Non-fatal: if DB is unavailable the alert still went out."""
    from datetime import timezone as _tz
    from hltrader.config import settings

    if not settings.HL_ALERT_TRACKING_ENABLED:
        return
    try:
        from hltrader.eval.outcomes import record_alert
        for c in candidates:
            record_alert(
                settings.pg_dsn,
                symbol=c.coin,
                alert_type=alert_type,
                score=c.composite,
                pump_score=c.pump_score,
                funding_score=c.funding_score,
                oi_score=c.oi_score,
                accel_score=c.accel_score,
                liquidity_score=c.liquidity_score,
                price_at_alert=c.price,
                alert_timestamp=datetime.now(_tz.utc),
                conviction_score=getattr(c, 'conviction_score', None),
                cluster_type=cluster_type,
            )
        console.print(f"[dim]alert_outcomes: recorded {len(candidates)} {alert_type} alert(s)[/dim]")
    except Exception as exc:
        console.print(f"[dim]alert_outcomes record failed (non-fatal): {exc}[/dim]")


def _print_candidates_table(candidates: list, title: str = "Short Candidates") -> None:
    """Print scored candidates as a Rich table."""
    table = Table(title=title)
    table.add_column("Type", style="dim")
    table.add_column("Coin", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("24h %", justify="right")
    table.add_column("Funding", justify="right")
    table.add_column("OI", justify="right")
    table.add_column("Volume", justify="right")

    type_colors = {"Crypto": "cyan", "Commodity": "yellow", "Stock": "magenta"}
    for c in candidates:
        tc = type_colors.get(c.asset_type, "white")
        table.add_row(
            f"[{tc}]{c.asset_type}[/{tc}]",
            c.coin,
            f"{c.composite:.0f}",
            f"${c.price:,.4f}" if c.price < 1 else f"${c.price:,.2f}",
            f"[green]+{c.pct_24h:.1f}%[/green]",
            f"[red]{c.funding_rate:.4%}[/red]",
            f"${c.open_interest:,.0f}",
            f"${c.volume_24h:,.0f}",
        )

    console.print(table)


def watch_loop(
    *,
    interval: int = 180,
    min_score: float = 10.0,
    min_volume: float = 1_000_000.0,
    min_pct_24h: float = 8.0,
    min_oi: float = 0.0,
    pump_cap: float = 25.0,
    pump_exp: float = 1.6,
    liquidity_percentile_keep: float = 0.70,
    liquidity_metric: str = "auto",
    cooldown: int = 3600,
    extreme_enabled: bool = True,
    extreme_pct_24h: float = 20.0,
    extreme_score_threshold: float = 45.0,
    extreme_max_items: int = 3,
    extreme_cooldown: int = 21600,
) -> None:
    """Blocking loop: fetch, score, alert, sleep. Runs until Ctrl-C."""
    from hltrader.config import settings

    # Phase 2: Set candle cache TTL
    set_cache_ttl(settings.HL_SCAN_CANDLE_CACHE_TTL)

    tg_ok = is_telegram_configured()
    console.print(f"[bold]Watcher started[/bold] — interval={interval}s, "
                  f"scan_floor={min_score}, alert_cap=45, cooldown={cooldown}s")
    console.print(f"  pump_cap={pump_cap}%, pump_exp={pump_exp}, "
                  f"min_pct={min_pct_24h}%, liq_keep={liquidity_percentile_keep:.0%}")
    console.print(f"  [blue]Phase 2: weights=P{settings.HL_SCAN_W_PUMP}/F{settings.HL_SCAN_W_FUNDING}"
                  f"/O{settings.HL_SCAN_W_OI}/A{settings.HL_SCAN_W_ACCEL}, "
                  f"pctl={settings.HL_SCAN_PERCENTILE_MODE}(pump={settings.HL_SCAN_PUMP_PERCENTILE_KEEP:.0%},"
                  f"score={settings.HL_SCAN_SCORE_PERCENTILE_KEEP:.0%})[/blue]")
    if extreme_enabled:
        console.print(f"  [yellow]EXTREME lane: pct>={extreme_pct_24h}% or score>{extreme_score_threshold}, "
                      f"max={extreme_max_items}, cooldown={extreme_cooldown}s[/yellow]")
    if settings.HL_REGIME_ENABLED:
        console.print(f"  [magenta]Phase 5 Regime: slope_up={settings.HL_REGIME_SLOPE_UP}, "
                      f"slope_down={settings.HL_REGIME_SLOPE_DOWN}, "
                      f"atr_expand={settings.HL_REGIME_ATR_EXPAND}, "
                      f"atr_contract={settings.HL_REGIME_ATR_CONTRACT}, "
                      f"conv_adjust={settings.HL_REGIME_CONVICTION_ADJUST}[/magenta]")
    if settings.HL_PUMP_FADE_ENABLED:
        console.print(f"  [cyan]Pump-Fade scoring: min_score={settings.HL_PUMP_FADE_MIN_SCORE}, "
                      f"cache_ttl={settings.HL_PUMP_FADE_CACHE_TTL}s[/cyan]")
        if settings.HL_TELEGRAM_CHAT_ID_TRADES:
            console.print(f"  [cyan]TRADES chat: configured (min_score={settings.HL_PUMP_FADE_TRADE_MIN_SCORE}, "
                          f"cooldown={settings.HL_PUMP_FADE_TRADE_COOLDOWN_MIN}min)[/cyan]")
        else:
            log.warning("Pump-Fade enabled but HL_TELEGRAM_CHAT_ID_TRADES is empty — trade alerts disabled")
            console.print("  [yellow]TRADES chat: not configured (HL_TELEGRAM_CHAT_ID_TRADES empty)[/yellow]")
    if settings.HL_ADAPTIVE_WEIGHTS_ENABLED:
        console.print(f"  [blue]Phase 6.1 Adaptive Weights: enabled "
                      f"(ema={settings.HL_ADAPTIVE_WEIGHTS_EMA_ALPHA}, "
                      f"delta_cap={settings.HL_ADAPTIVE_WEIGHTS_MAX_DAILY_DELTA}, "
                      f"min_global={settings.HL_ADAPTIVE_WEIGHTS_MIN_GLOBAL})[/blue]")
    console.print(f"Telegram: {'[green]configured[/green]' if tg_ok else '[yellow]not configured[/yellow]'}")
    console.print("[dim]Press Ctrl-C to stop[/dim]\n")

    try:
        while True:
            try:
                now = datetime.now().strftime("%H:%M:%S")
                console.print(f"[dim][{now}] Scanning...[/dim]")

                snapshots = _fetch_24h_snapshots()

                # Phase 5: Market Regime
                regime_result = None
                if settings.HL_REGIME_ENABLED:
                    try:
                        from hltrader.analysis.regime import compute_regime, store_regime_snapshot
                        regime_result = compute_regime(
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
                        if not regime_result.cached and settings.HL_REGIME_STORE_ENABLED:
                            store_regime_snapshot(settings.pg_dsn, regime_result)
                        console.print(f"[dim]regime: {regime_result.regime_code} | risk: {regime_result.risk_state}"
                                      f"{' (cached)' if regime_result.cached else ''}[/dim]")
                    except Exception as exc:
                        console.print(f"[dim]regime failed (non-fatal): {exc}[/dim]")

                # Phase 2: Pre-filter for candle fetch (only coins passing basic gates)
                _pf_coins = [
                    s["coin"] for s in snapshots
                    if s["pct_24h"] >= min_pct_24h
                    and s["volume_24h"] >= min_volume
                ][:50]  # cap API calls

                # Fetch 1h/4h returns for acceleration scoring
                returns_cache = {}
                if settings.HL_SCAN_W_ACCEL > 0 and _pf_coins:
                    try:
                        returns_cache = build_returns_cache(_pf_coins)
                        log.info("candle_fetch: %d/%d coins got returns",
                                 sum(1 for v in returns_cache.values()
                                     if v.get("ret_1h") is not None),
                                 len(_pf_coins))
                    except Exception as exc:
                        log.warning("candle_fetch failed (non-fatal): %s", exc)

                # Phase 6/6.1: Load adaptive weights (override static .env)
                _aw_pump = settings.HL_SCAN_W_PUMP
                _aw_funding = settings.HL_SCAN_W_FUNDING
                _aw_oi = settings.HL_SCAN_W_OI
                _aw_accel = settings.HL_SCAN_W_ACCEL
                _aw_source = "STATIC"
                if settings.HL_ADAPTIVE_WEIGHTS_ENABLED:
                    try:
                        from hltrader.analysis.adaptive_weights import get_weights_for_regime
                        _regime_code = regime_result.regime_code if regime_result else None
                        _ws = get_weights_for_regime(
                            settings.pg_dsn,
                            regime_code=_regime_code,
                            static_weights={
                                "w_pump": settings.HL_SCAN_W_PUMP,
                                "w_oi": settings.HL_SCAN_W_OI,
                                "w_funding": settings.HL_SCAN_W_FUNDING,
                                "w_accel": settings.HL_SCAN_W_ACCEL,
                            },
                        )
                        _aw_pump = _ws.weights["w_pump"]
                        _aw_funding = _ws.weights["w_funding"]
                        _aw_oi = _ws.weights["w_oi"]
                        _aw_accel = _ws.weights["w_accel"]
                        _aw_source = _ws.source
                        console.print(f"[dim]weights: {_ws.source}"
                                      f"{' (' + _ws.regime_code + ')' if _ws.regime_code else ''}"
                                      f" P{_aw_pump:.1f}/F{_aw_funding:.1f}"
                                      f"/O{_aw_oi:.1f}/A{_aw_accel:.1f}"
                                      f" n={_ws.sample_size}[/dim]")
                    except Exception as exc:
                        console.print(f"[dim]adaptive_weights load failed (non-fatal): {exc}[/dim]")

                candidates = score_short_candidates(
                    snapshots,
                    min_score=min_score,
                    min_volume=min_volume,
                    min_pct_24h=min_pct_24h,
                    min_oi=min_oi,
                    pump_cap=pump_cap,
                    pump_exp=pump_exp,
                    liquidity_percentile_keep=liquidity_percentile_keep,
                    liquidity_metric=liquidity_metric,
                    # Phase 2 params
                    w_pump=_aw_pump,
                    w_funding=_aw_funding,
                    w_oi=_aw_oi,
                    w_accel=_aw_accel,
                    accel_a0=settings.HL_SCAN_ACCEL_A0,
                    accel_a1=settings.HL_SCAN_ACCEL_A1,
                    pump_percentile_keep=settings.HL_SCAN_PUMP_PERCENTILE_KEEP,
                    score_percentile_keep=settings.HL_SCAN_SCORE_PERCENTILE_KEEP,
                    percentile_mode=settings.HL_SCAN_PERCENTILE_MODE,
                    returns_cache=returns_cache,
                )

                # Structured log for observability
                stats = {
                    "total_assets": len(snapshots),
                    "passed_volume_pct": len(_pf_coins),
                    "candle_fetched": len(returns_cache),
                    "passed_scoring": len(candidates),
                    "weights_source": _aw_source,
                }

                if candidates:
                    # Phase 4: Compute conviction scores
                    if settings.HL_CONVICTION_ENABLED:
                        from hltrader.analysis.conviction import compute_conviction
                        _conviction_data = {}
                        for c in candidates:
                            try:
                                _rc = regime_result.regime_code if regime_result else ""
                                cv = compute_conviction(
                                    symbol=c.coin,
                                    alert_type="PENDING",
                                    composite_score=c.composite,
                                    pump_score=c.pump_score,
                                    funding_score=c.funding_score,
                                    oi_score=c.oi_score,
                                    accel_score=c.accel_score,
                                    liquidity=c.liquidity_score,
                                    alert_timestamp=datetime.now(),
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
                                    # Phase 7: Token memory as final refinement
                                    regime_code=_rc,
                                    token_memory_enabled=settings.HL_TOKEN_MEMORY_CONVICTION_ENABLED and settings.HL_TOKEN_MEMORY_ENABLED,
                                    token_memory_max_boost=settings.HL_TOKEN_MEMORY_MAX_BOOST,
                                    token_memory_max_penalty=settings.HL_TOKEN_MEMORY_MAX_PENALTY,
                                    token_memory_confidence_min=settings.HL_TOKEN_MEMORY_CONFIDENCE_MIN,
                                )
                                c.conviction_score = cv["conviction"]
                                c.conviction_reasons = cv.get("reasons", [])
                                c.conviction_allowed = cv.get("allowed", True)
                                _conviction_data[c.coin] = cv
                            except Exception as exc:
                                log.warning("conviction failed for %s (non-fatal): %s", c.coin, exc)
                                c.conviction_score = None

                        # Shadow mode: log but don't block
                        if settings.HL_CONVICTION_SHADOW:
                            would_block = sum(
                                1 for c in candidates
                                if c.conviction_score is not None and c.conviction_score < settings.HL_CONVICTION_MIN
                            )
                            if would_block:
                                console.print(f"[dim]conviction shadow: {would_block} candidate(s) would be blocked (< {settings.HL_CONVICTION_MIN})[/dim]")
                            stats["conviction_would_block"] = would_block
                            stats["conviction_suppressed"] = 0
                        else:
                            # Block mode: filter by conviction threshold
                            pre_count = len(candidates)
                            candidates = [
                                c for c in candidates
                                if c.conviction_score is None or c.conviction_score >= settings.HL_CONVICTION_MIN
                            ]
                            suppressed = pre_count - len(candidates)
                            if suppressed:
                                console.print(f"[dim]conviction: suppressed {suppressed} candidate(s) below {settings.HL_CONVICTION_MIN}[/dim]")
                            stats["conviction_suppressed"] = suppressed

                        # Phase 5: Regime-based conviction adjustment
                        if regime_result and settings.HL_REGIME_CONVICTION_ADJUST:
                            for c in candidates:
                                if c.conviction_score is not None:
                                    if regime_result.risk_state == "risk_on":
                                        c.conviction_score = max(0.0, c.conviction_score - settings.HL_REGIME_CONVICTION_UP_PENALTY)
                                    elif regime_result.risk_state == "risk_off":
                                        c.conviction_score = min(100.0, c.conviction_score + settings.HL_REGIME_CONVICTION_DOWN_BOOST)

                    # === Phase 5.3: Signal Clustering ===
                    clusters = []
                    clustered_coins = set()
                    if settings.HL_CLUSTER_ENABLED:
                        try:
                            clusters, candidates = detect_clusters(
                                candidates,
                                min_cluster_size=settings.HL_CLUSTER_MIN_SIZE,
                                conviction_boost=settings.HL_CLUSTER_CONVICTION_BOOST,
                            )
                            stats["clusters_detected"] = len(clusters)
                            cluster_alerts_sent = 0
                            for cluster in clusters:
                                alertable = [c for c in cluster.candidates if _should_alert(c.coin, cooldown)]
                                if len(alertable) < settings.HL_CLUSTER_MIN_SIZE:
                                    candidates.extend(alertable)
                                    continue
                                display = alertable[:settings.HL_CLUSTER_MAX_PER_MESSAGE]
                                msg = format_cluster_alert(cluster, display, regime=regime_result)
                                if msg and tg_ok:
                                    try:
                                        send_telegram(msg)
                                        console.print(
                                            f"[magenta]CLUSTER alert sent: {cluster.label} "
                                            f"({len(display)} coins)[/magenta]"
                                        )
                                    except Exception as exc:
                                        console.print(f"[red]Cluster Telegram error:[/red] {exc}")
                                for c in display:
                                    _record_alert(c.coin)
                                    clustered_coins.add(c.coin)
                                _record_alert_outcomes(display, "CLUSTER", cluster_type=cluster.sector)
                                cluster_alerts_sent += 1
                            stats["cluster_alerts_sent"] = cluster_alerts_sent
                        except Exception:
                            pass  # fail-open, candidates unchanged

                    _print_candidates_table(candidates)

                    # Classify into normal vs extreme
                    normal_candidates, extreme_candidates = classify_extreme_candidates(
                        candidates,
                        score_threshold=extreme_score_threshold,
                        pct_threshold=extreme_pct_24h,
                    )

                    # === NORMAL LANE ===
                    # Apply existing MAX_ALERT_SCORE filter (defense in depth)
                    alert_candidates = filter_alert_candidates(normal_candidates)[:5]
                    new_candidates = [
                        c for c in alert_candidates if _should_alert(c.coin, cooldown)
                    ]

                    if new_candidates and tg_ok:
                        msg = format_short_alert_batch(new_candidates, regime=regime_result)
                        if msg:
                            try:
                                send_telegram(msg)
                                console.print(
                                    f"[green]Telegram alert sent for "
                                    f"{len(new_candidates)} coin(s)[/green]"
                                )
                                _record_alert_outcomes(new_candidates, "NORMAL")
                            except Exception as exc:
                                console.print(f"[red]Telegram error:[/red] {exc}")

                    for c in new_candidates:
                        _record_alert(c.coin)

                    stats["normal_alert_count"] = len(new_candidates)

                    if not new_candidates:
                        console.print("[dim]Normal: all candidates on cooldown or filtered[/dim]")

                    # === EXTREME LANE ===
                    stats["extreme_candidate_count"] = len(extreme_candidates)
                    extreme_alerted = 0

                    if extreme_enabled and extreme_candidates and tg_ok:
                        # Apply cooldown and max items
                        new_extreme = [
                            c for c in extreme_candidates
                            if _should_alert_extreme(c.coin, extreme_cooldown)
                        ][:extreme_max_items]

                        # Pre-compute pump-fade scores for extreme candidates
                        fade_data: dict = {}
                        if settings.HL_PUMP_FADE_ENABLED and new_extreme:
                            from hltrader.scan.pump_fade import score_pump_fade
                            for c in new_extreme:
                                try:
                                    fade_data[c.coin] = score_pump_fade(
                                        c.coin,
                                        {"pct_24h": c.pct_24h, "composite": c.composite,
                                         "funding_rate": c.funding_rate},
                                        pg_dsn=settings.pg_dsn_coinbase,
                                        cache_ttl=settings.HL_PUMP_FADE_CACHE_TTL,
                                        hl_timeout=settings.HL_PUMP_FADE_TIMEOUT_S,
                                    )
                                except Exception as exc:
                                    console.print(f"[dim]pump_fade failed for {c.coin} (non-fatal): {exc}[/dim]")

                        if new_extreme:
                            msg = format_extreme_pump_batch(
                                new_extreme,
                                score_threshold=extreme_score_threshold,
                                pct_threshold=extreme_pct_24h,
                                fade_data=fade_data,
                                regime=regime_result,
                            )
                            if msg:
                                try:
                                    send_telegram(msg)
                                    console.print(
                                        f"[yellow]EXTREME PUMP alert sent for "
                                        f"{len(new_extreme)} coin(s)[/yellow]"
                                    )
                                except Exception as exc:
                                    console.print(f"[red]Extreme Telegram error:[/red] {exc}")

                            for c in new_extreme:
                                _record_extreme_alert(c.coin)
                            extreme_alerted = len(new_extreme)

                            # Record alert outcomes for performance tracking
                            _record_alert_outcomes(new_extreme, "EXTREME")

                            # Record cross-dedupe keys for inter-service coordination
                            _record_xdedupe_keys(new_extreme)

                            # === TRADES ROUTING ===
                            if settings.HL_PUMP_FADE_ENABLED and settings.HL_TELEGRAM_CHAT_ID_TRADES:
                                trade_cooldown_secs = settings.HL_PUMP_FADE_TRADE_COOLDOWN_MIN * 60
                                for c in new_extreme:
                                    fd = fade_data.get(c.coin)
                                    if fd and should_send_trade_alert(fd, settings):
                                        if _should_alert_trade(c.coin, trade_cooldown_secs):
                                            msg = format_trade_alert(c.coin, c, fd)
                                            try:
                                                send_telegram(msg, chat_id=settings.HL_TELEGRAM_CHAT_ID_TRADES)
                                                _record_trade_alert(c.coin)
                                                console.print(f"[cyan]TRADE alert sent for {c.coin}[/cyan]")
                                            except Exception as exc:
                                                console.print(f"[dim]Trade alert failed (non-fatal): {exc}[/dim]")
                                        else:
                                            console.print(f"[dim]TRADE {c.coin}: on cooldown[/dim]")

                    stats["extreme_alert_count"] = extreme_alerted

                    if extreme_candidates and not extreme_alerted:
                        console.print("[dim]Extreme: all on cooldown or disabled[/dim]")

                else:
                    stats["normal_alert_count"] = 0
                    stats["extreme_candidate_count"] = 0
                    stats["extreme_alert_count"] = 0
                    console.print("[dim]No short candidates this cycle[/dim]")

                if regime_result:
                    stats["regime"] = regime_result.regime_code
                    stats["risk_state"] = regime_result.risk_state

                # Structured log line
                log.info("scan_cycle: %s", " | ".join(f"{k}={v}" for k, v in stats.items()))

            except Exception as exc:
                console.print(f"[red]Scan error:[/red] {exc}")

            time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Watcher stopped[/yellow]")
