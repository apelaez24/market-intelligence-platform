"""Telegram notification helpers."""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

from hltrader.config import settings
from hltrader.scan.scorer import ShortCandidate


def is_telegram_configured() -> bool:
    """Return True if both bot token and chat ID are set."""
    return bool(settings.HL_TELEGRAM_BOT_TOKEN and settings.HL_TELEGRAM_CHAT_ID)


def send_telegram(message: str, *, chat_id: str | None = None, parse_mode: str = "HTML") -> dict:
    """Send a message via Telegram Bot API.

    If *chat_id* is None the default FEED chat is used (backward-compatible).
    """
    token = settings.HL_TELEGRAM_BOT_TOKEN
    resolved_chat_id = chat_id or settings.feed_chat_id
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": resolved_chat_id, "text": message, "parse_mode": parse_mode},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def format_short_alert_batch(candidates: list[ShortCandidate], *, regime: object | None = None) -> str:
    """Format a batch of short candidates into a single HTML Telegram message.

    Defense-in-depth: silently drops any candidate with composite > 45.
    """
    from hltrader.scan.scorer import MAX_ALERT_SCORE
    safe = [c for c in candidates if c.composite <= MAX_ALERT_SCORE]
    if len(safe) < len(candidates):
        dropped = len(candidates) - len(safe)
        log.warning("Defense-in-depth: dropped %d candidate(s) with score > %d in notify", dropped, MAX_ALERT_SCORE)
    if not safe:
        return ""
    lines = ["<b>Short Candidates Detected</b>", ""]
    if regime is not None:
        lines.append(f"Regime: {regime.regime_code} | Risk: {regime.risk_state}")
        lines.append("")
    for c in safe:
        type_emoji = {"Crypto": "\U0001f4b0", "Commodity": "\U0001f947", "Stock": "\U0001f4c8"}.get(c.asset_type, "")
        tier_label = f" Tier {c.tier}" if getattr(c, "tier", "") else ""
        lines.append(
            f"{type_emoji} <b>{c.coin}</b> [{c.asset_type}]{tier_label}  Score: {c.composite:.0f}/100\n"
            f"  Price: ${c.price:,.4f}  |  24h: +{c.pct_24h:.1f}%\n"
            f"  Funding: {c.funding_rate:.4%}  |  OI: ${c.open_interest:,.0f}\n"
            f"  Vol: ${c.volume_24h:,.0f}"
        )
        # Phase 2: Show 1h/4h returns + acceleration if available
        ret_parts = []
        if getattr(c, "ret_1h", None) is not None:
            ret_parts.append(f"1h: {c.ret_1h:+.1f}%")
        if getattr(c, "ret_4h", None) is not None:
            ret_parts.append(f"4h: {c.ret_4h:+.1f}%")
        if getattr(c, "accel_score", 0) > 0:
            ret_parts.append(f"Accel: {c.accel_score:.2f}")
        if ret_parts:
            lines.append(f"  {' | '.join(ret_parts)}")
        # Phase 4: Conviction score
        if getattr(c, "conviction_score", None) is not None:
            from hltrader.analysis.conviction import conviction_tier
            ct = conviction_tier(c.conviction_score)
            tier_str = f" Tier {ct}" if ct else ""
            lines.append(f"  Conviction: {c.conviction_score:.0f}/100{tier_str}")
            # Phase 4.1: Top 2 reasons
            cv_reasons = getattr(c, "conviction_reasons", None)
            cv_allowed = getattr(c, "conviction_allowed", None)
            if cv_reasons:
                label = "Why allowed" if cv_allowed else "Why blocked"
                lines.append(f"  {label}: {', '.join(cv_reasons)}")
        # Phase 7: Token personality line
        try:
            from hltrader.analysis.token_memory import format_personality_line
            from hltrader.config import settings as _tm_s
            if _tm_s.HL_TOKEN_MEMORY_ENABLED:
                pline = format_personality_line(_tm_s.pg_dsn, c.coin)
                if pline:
                    lines.append(f"  {pline}")
        except Exception:
            pass
        lines.append("")
    return "\n".join(lines).strip()


def format_extreme_pump_batch(
    candidates: list[ShortCandidate],
    *,
    score_threshold: float = 45.0,
    pct_threshold: float = 20.0,
    fade_data: dict | None = None,
    regime: object | None = None,
) -> str:
    """Format extreme pump candidates into a separate Telegram message.

    No score cap filtering — these are intentionally high-score/high-pump items
    that were excluded from the normal lane.
    """
    if not candidates:
        return ""
    lines = ["\u26a0\ufe0f <b>EXTREME PUMP WATCH</b>", ""]
    if regime is not None:
        lines.append(f"Regime: {regime.regime_code} | Risk: {regime.risk_state}")
        lines.append("")
    for c in candidates:
        type_emoji = {"Crypto": "\U0001f4b0", "Commodity": "\U0001f947", "Stock": "\U0001f4c8"}.get(c.asset_type, "")

        # Build "Why flagged" line
        reasons: list[str] = []
        if c.composite > score_threshold:
            reasons.append(f"score>{score_threshold:.0f}")
        if c.pct_24h >= pct_threshold:
            reasons.append(f"pct>={pct_threshold:.0f}%")
        why = " + ".join(reasons) if reasons else "extreme conditions"

        # Phase 2: Build 1h/4h/accel line for extreme candidates
        _ret_parts = []
        if getattr(c, "ret_1h", None) is not None:
            _ret_parts.append(f"1h: {c.ret_1h:+.1f}%")
        if getattr(c, "ret_4h", None) is not None:
            _ret_parts.append(f"4h: {c.ret_4h:+.1f}%")
        if getattr(c, "accel_score", 0) > 0:
            _ret_parts.append(f"Accel: {c.accel_score:.2f}")
        _ret_line = f"\n  {' | '.join(_ret_parts)}" if _ret_parts else ""

        lines.append(
            f"{type_emoji} <b>{c.coin}</b> [{c.asset_type}]  Score: {c.composite:.0f}/100\n"
            f"  Price: ${c.price:,.4f}  |  24h: +{c.pct_24h:.1f}%\n"
            f"  Funding: {c.funding_rate:.4%}  |  OI: ${c.open_interest:,.0f}\n"
            f"  Vol: ${c.volume_24h:,.0f}{_ret_line}\n"
            f"  Why flagged: {why}\n"
            f"  Playbook: wait for rejection candle, check funding flip"
        )
        # Phase 4: Conviction score for extreme lane
        if getattr(c, "conviction_score", None) is not None:
            from hltrader.analysis.conviction import conviction_tier
            ct = conviction_tier(c.conviction_score)
            tier_str = f" Tier {ct}" if ct else ""
            lines.append(f"  Conviction: {c.conviction_score:.0f}/100{tier_str}")
            # Phase 4.1: Top 2 reasons
            cv_reasons = getattr(c, "conviction_reasons", None)
            cv_allowed = getattr(c, "conviction_allowed", None)
            if cv_reasons:
                label = "Why allowed" if cv_allowed else "Why blocked"
                lines.append(f"  {label}: {', '.join(cv_reasons)}")

        # Pump-Fade addendum (if available and meets threshold)
        if fade_data and c.coin in fade_data:
            fd = fade_data[c.coin]
            try:
                from hltrader.config import settings as _s
                min_score = _s.HL_PUMP_FADE_MIN_SCORE
            except Exception:
                min_score = 60
            if fd["score"] >= min_score:
                rollover_icon = "\u2705" if fd["rollover"] else "\u274c"
                lines.append(
                    f"  \U0001f4c9 Fade: {fd['score']}/100 "
                    f"({fd['confidence']:.0%} conf) "
                    f"Rollover: {rollover_icon}"
                )
                if fd.get("rsi_14_5m") is not None:
                    rsi_1h_str = f" / {fd['rsi_14_1h']:.0f} (1h)" if fd.get("rsi_14_1h") else ""
                    lines.append(f"  RSI: {fd['rsi_14_5m']:.0f} (5m){rsi_1h_str}")
                if fd.get("bb_pct_b") is not None:
                    lines.append(f"  BB: pct_b={fd['bb_pct_b']:.2f}  width={fd['bb_bandwidth']:.3f}")
                for r in fd["reasons"][:2]:
                    lines.append(f"  \u2192 {r}")
                if fd["risk_flags"]:
                    lines.append(f"  \u26a0\ufe0f {fd['risk_flags'][0]}")

        # Phase 7: Token personality line
        try:
            from hltrader.analysis.token_memory import format_personality_line
            from hltrader.config import settings as _tm_s2
            if _tm_s2.HL_TOKEN_MEMORY_ENABLED:
                pline = format_personality_line(_tm_s2.pg_dsn, c.coin)
                if pline:
                    lines.append(f"  {pline}")
        except Exception:
            pass
        lines.append(f"  <i>Do NOT front-run; wait for reversal confirmation.</i>")
        lines.append("")
    return "\n".join(lines).strip()



def should_send_trade_alert(fade: dict, settings) -> bool:
    """Pure routing decision: should this fade result go to the TRADES chat?

    Returns True only when ALL conditions are met:
    - Pump-Fade is enabled
    - TRADES chat ID is configured
    - Fade score >= trade min score threshold
    - RSI rollover confirmed
    """
    return (
        settings.HL_PUMP_FADE_ENABLED
        and bool(settings.HL_TELEGRAM_CHAT_ID_TRADES)
        and fade.get("score", 0) >= settings.HL_PUMP_FADE_TRADE_MIN_SCORE
        and fade.get("rollover") is True
    )


def format_trade_alert(coin: str, candidate, fade: dict) -> str:
    """Format a concise standalone trade alert for the TRADES chat."""
    score = fade.get("score", 0)
    conf = fade.get("confidence", 0.0)
    rollover_icon = "\u2705" if fade.get("rollover") else "\u274c"
    rsi_5m = fade.get("rsi_14_5m")
    rsi_1h = fade.get("rsi_14_1h")
    bb_pct_b = fade.get("bb_pct_b")
    bb_bw = fade.get("bb_bandwidth")

    lines = [
        f"\U0001f6a8 <b>SHORT CANDIDATE (Pump Fade) \u2014 {coin}</b>",
        f"Fade Score: {score}/100 ({conf:.0%} conf)",
    ]

    rsi_parts = f"RSI(5m): {rsi_5m:.0f}" if rsi_5m is not None else ""
    if rsi_1h is not None:
        rsi_parts += f"  RSI(1h): {rsi_1h:.0f}"
    lines.append(f"Rollover: {rollover_icon}  {rsi_parts}".rstrip())

    if bb_pct_b is not None:
        lines.append(f"BB pct_b: {bb_pct_b:.2f}  width: {bb_bw:.3f}")

    for r in fade.get("reasons", [])[:3]:
        lines.append(f"\u2192 {r}")

    for rf in fade.get("risk_flags", [])[:2]:
        lines.append(f"\u26a0\ufe0f {rf}")

    return "\n".join(lines)


def format_cluster_alert(cluster, candidates, *, regime=None) -> str:
    """Format a sector cluster into a single consolidated Telegram alert.

    Duck-types cluster (.label, .avg_pct_24h, .avg_composite, .sector)
    and regime (.regime_code, .risk_state).
    """
    if not candidates:
        return ""
    label = getattr(cluster, "label", "Unknown")
    avg_pct = getattr(cluster, "avg_pct_24h", 0.0)
    avg_score = getattr(cluster, "avg_composite", 0.0)

    lines = [
        f"\U0001f517 <b>{label.upper()} SECTOR ROTATION ({len(candidates)} moving)</b>",
        "",
    ]

    if regime is not None:
        rc = getattr(regime, "regime_code", "UNKNOWN")
        rs = getattr(regime, "risk_state", "unknown")
        lines.append(f"Regime: {rc} | Risk: {rs}")
        lines.append("")

    lines.append(f"Avg pump: +{avg_pct:.1f}% | Avg score: {avg_score:.0f}")
    lines.append("")

    for c in candidates:
        type_emoji = {"Crypto": "\U0001f4b0", "Commodity": "\U0001f947", "Stock": "\U0001f4c8"}.get(
            getattr(c, "asset_type", ""), "")
        conv = getattr(c, "conviction_score", None)
        conv_str = f"  Conv: {conv:.0f}" if conv is not None else ""
        tier = getattr(c, "tier", "")
        tier_str = f" Tier {tier}" if tier else ""
        lines.append(
            f"{type_emoji} <b>{c.coin}</b>  +{c.pct_24h:.1f}%  Score: {c.composite:.0f}{conv_str}{tier_str}"
        )

    # Show boost amount
    boost = getattr(cluster, "_conviction_boost", None)
    sector = getattr(cluster, "sector", "")
    lines.append("")
    lines.append(f"Cluster boost applied ({sector})")

    return "\n".join(lines)
