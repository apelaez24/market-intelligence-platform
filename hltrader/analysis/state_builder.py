"""Phase 5.1: Market State Snapshot Engine.

Builds a unified market_state.json + market_state.md from local DB data.
Atomic write, no secrets, degraded mode if DB unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

log = logging.getLogger(__name__)

STATE_DIR = Path("/home/pi/operator/state")
STATE_JSON = STATE_DIR / "market_state.json"
STATE_MD = STATE_DIR / "market_state.md"


# ── Helpers ──────────────────────────────────────────────────

def _ensure_dir() -> None:
    """Create state directory with 700 perms if missing."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(str(STATE_DIR), 0o700)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a temp file then rename (atomic on same FS)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.chmod(tmp, 0o600)
        os.rename(tmp, str(path))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_connect(pg_dsn: str):
    """Connect to DB. Returns conn or None."""
    try:
        return psycopg2.connect(pg_dsn)
    except Exception as exc:
        log.warning("DB connect failed: %s", exc)
        return None


# ── Data Fetchers ────────────────────────────────────────────

def fetch_regime(pg_dsn: str) -> dict | None:
    """Fetch newest regime snapshot from last 30 minutes."""
    conn = _safe_connect(pg_dsn)
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT regime_code, btc_trend, vol_state, risk_state,
                          btc_slope_4h, atr_ratio, breadth_pump10_pct,
                          breadth_dump10_pct, snapshot_ts
                   FROM market_regime_snapshots
                   WHERE snapshot_ts >= now() - interval '60 minutes'
                   ORDER BY snapshot_ts DESC LIMIT 1"""
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "regime_code": row[0],
            "btc_trend": row[1],
            "vol_state": row[2],
            "risk_state": row[3],
            "metrics": {
                "btc_slope_4h": row[4],
                "atr_ratio": row[5],
                "breadth_pump10_pct": row[6],
                "breadth_dump10_pct": row[7],
            },
        }
    except Exception as exc:
        log.warning("fetch_regime failed: %s", exc)
        return None
    finally:
        conn.close()


def fetch_alerts(pg_dsn: str) -> dict:
    """Fetch alert counts and top active alerts from last 24h."""
    result = {
        "last_24h_counts": {"NORMAL": 0, "EXTREME": 0},
        "shadow_blocked_24h": 0,
        "top_active": [],
    }
    conn = _safe_connect(pg_dsn)
    if not conn:
        return result
    try:
        with conn.cursor() as cur:
            # Counts by type
            cur.execute(
                """SELECT alert_type, COUNT(*)
                   FROM alert_outcomes
                   WHERE alert_timestamp >= now() - interval '24 hours'
                   GROUP BY alert_type"""
            )
            for atype, cnt in cur.fetchall():
                result["last_24h_counts"][atype] = cnt

            # Shadow blocked count (conviction < 55)
            try:
                cur.execute(
                    """SELECT COUNT(*)
                       FROM alert_outcomes
                       WHERE alert_timestamp >= now() - interval '24 hours'
                         AND conviction_score IS NOT NULL
                         AND conviction_score < 55"""
                )
                result["shadow_blocked_24h"] = cur.fetchone()[0]
            except Exception:
                pass

            # Top active alerts
            cur.execute(
                """SELECT symbol, alert_type, score, conviction_score,
                          CASE
                              WHEN conviction_score >= 75 THEN 'A'
                              WHEN conviction_score >= 60 THEN 'B'
                              WHEN conviction_score >= 55 THEN 'C'
                              ELSE '-'
                          END AS tier,
                          price_at_alert,
                          pump_score, funding_score, oi_score,
                          alert_timestamp
                   FROM alert_outcomes
                   WHERE alert_timestamp >= now() - interval '24 hours'
                   ORDER BY COALESCE(conviction_score, score) DESC
                   LIMIT 5"""
            )
            for row in cur.fetchall():
                result["top_active"].append({
                    "symbol": row[0],
                    "lane": row[1],
                    "score": round(float(row[2]), 1) if row[2] else 0,
                    "conviction": round(float(row[3]), 1) if row[3] else None,
                    "tier": row[4],
                    "price": round(float(row[5]), 4) if row[5] else 0,
                    "ts": row[9].isoformat() if row[9] else None,
                })
    except Exception as exc:
        log.warning("fetch_alerts failed: %s", exc)
    finally:
        conn.close()
    return result


def fetch_performance(pg_dsn: str) -> dict:
    """Fetch 7-day performance by tier and by regime."""
    result = {
        "window_days": 7,
        "tiers": {},
        "by_regime": [],
    }
    conn = _safe_connect(pg_dsn)
    if not conn:
        return result
    try:
        with conn.cursor() as cur:
            # Tier breakdown
            cur.execute(
                """SELECT
                       CASE
                           WHEN conviction_score >= 75 THEN 'A'
                           WHEN conviction_score >= 60 THEN 'B'
                           WHEN conviction_score >= 55 THEN 'C'
                           ELSE '<55'
                       END AS tier,
                       COUNT(*) AS cnt,
                       COUNT(*) FILTER (WHERE eval_1h_return < 0) AS win_1h,
                       COUNT(*) FILTER (WHERE evaluated_1h) AS eval_1h,
                       COUNT(*) FILTER (WHERE eval_4h_return < 0) AS win_4h,
                       COUNT(*) FILTER (WHERE evaluated_4h) AS eval_4h,
                       COUNT(*) FILTER (WHERE eval_24h_return < 0) AS win_24h,
                       COUNT(*) FILTER (WHERE evaluated_24h) AS eval_24h,
                       AVG(eval_24h_return) FILTER (WHERE evaluated_24h) AS avg_24h
                   FROM alert_outcomes
                   WHERE alert_timestamp >= now() - interval '7 days'
                   GROUP BY tier
                   ORDER BY tier"""
            )
            for tier, cnt, w1, e1, w4, e4, w24, e24, avg24 in cur.fetchall():
                result["tiers"][tier] = {
                    "count": cnt,
                    "win_1h": round(w1 / e1, 3) if e1 else None,
                    "win_4h": round(w4 / e4, 3) if e4 else None,
                    "win_24h": round(w24 / e24, 3) if e24 else None,
                    "avg_24h": round(float(avg24), 2) if avg24 is not None else None,
                }

            # By regime (nearest-back join)
            cur.execute(
                """SELECT r.regime_code,
                          COUNT(*) AS cnt,
                          AVG(a.eval_24h_return) FILTER (WHERE a.evaluated_24h) AS avg_24h,
                          COUNT(*) FILTER (WHERE a.eval_24h_return < 0 AND a.evaluated_24h) AS win_24h,
                          COUNT(*) FILTER (WHERE a.evaluated_24h) AS eval_24h
                   FROM alert_outcomes a
                   LEFT JOIN LATERAL (
                       SELECT regime_code
                       FROM market_regime_snapshots
                       WHERE snapshot_ts <= a.alert_timestamp
                       ORDER BY snapshot_ts DESC
                       LIMIT 1
                   ) r ON true
                   WHERE a.alert_timestamp >= now() - interval '7 days'
                     AND r.regime_code IS NOT NULL
                   GROUP BY r.regime_code
                   ORDER BY cnt DESC
                   LIMIT 6"""
            )
            for regime_code, cnt, avg24, w24, e24 in cur.fetchall():
                result["by_regime"].append({
                    "regime_code": regime_code,
                    "count": cnt,
                    "avg_24h": round(float(avg24), 2) if avg24 is not None else None,
                    "win_24h": round(w24 / e24, 3) if e24 else None,
                })
    except Exception as exc:
        log.warning("fetch_performance failed: %s", exc)
    finally:
        conn.close()
    return result


def fetch_geo(pg_dsn: str) -> dict:
    """Fetch active high-severity geo events from last 6h."""
    result = {"active": False, "top": []}
    conn = _safe_connect(pg_dsn)
    if not conn:
        return result
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT severity, headline, created_at, source
                   FROM geo_events
                   WHERE severity >= 80
                     AND created_at >= now() - interval '6 hours'
                   ORDER BY severity DESC
                   LIMIT 3"""
            )
            for sev, headline, ts, source in cur.fetchall():
                result["top"].append({
                    "severity": sev,
                    "headline": headline,
                    "ts": ts.isoformat() if ts else None,
                    "source": source or "guardian",
                })
        result["active"] = len(result["top"]) > 0
    except Exception as exc:
        log.warning("fetch_geo failed: %s", exc)
    finally:
        conn.close()
    return result


def check_service_status(service_name: str) -> str:
    """Check systemd service status. Returns 'active' or 'inactive'."""
    try:
        r = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "inactive"
    except Exception:
        return "unknown"


def check_timer_status(timer_name: str) -> str:
    """Check systemd timer status. Returns 'active' or 'inactive'.

    For oneshot services triggered by timers, the service itself shows
    'inactive' between runs. Check the timer instead.
    """
    try:
        r = subprocess.run(
            ["systemctl", "is-active", timer_name],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "inactive"
    except Exception:
        return "unknown"


def fetch_system() -> dict:
    """Check core service statuses."""
    return {
        "services": {
            "hltrader_watcher": check_service_status("hltrader-watcher"),
            "telegram_bot": check_service_status("telegram-bot"),
            "hltrader_evaluator": check_timer_status("hltrader-evaluator.timer"),
            "hltrader_state": check_timer_status("hltrader-state.timer"),
        }
    }


# ── Sector Classification ────────────────────────────────────

# Sector data imported from canonical source (cluster.py)
from hltrader.scan.cluster import SECTOR_MAP as _SECTOR_MAP, SECTOR_LABELS as _SECTOR_LABELS


# ── Narrative Engine ─────────────────────────────────────────

def build_narrative(regime: dict | None, alerts: dict, geo: dict) -> dict:
    """Build market narrative from regime, alerts, and geo data.

    Deterministic — no LLM, no external APIs. Derives a human-readable
    narrative structure from the current state data.
    """
    narrative = {
        "market_driver": "mixed",
        "sentiment": "neutral",
        "breadth_strength": "normal",
        "volatility_context": "stable",
        "sector_heat": [],
        "geo_driver": None,
        "summary": "",
    }

    # ── Sector heat: which sectors are active in top alerts ──
    sector_counts: dict[str, int] = {}
    for alert in alerts.get("top_active", []):
        sym = alert.get("symbol", "")
        sector = _SECTOR_MAP.get(sym, "other")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Sort by count descending, take top 3
    heat = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    narrative["sector_heat"] = [
        {"sector": s, "label": _SECTOR_LABELS.get(s, s), "count": c}
        for s, c in heat
    ]

    # Market driver = top sector
    if heat:
        narrative["market_driver"] = _SECTOR_LABELS.get(heat[0][0], heat[0][0])
    else:
        narrative["market_driver"] = "quiet"

    # ── Sentiment from regime ──
    if regime:
        risk = regime.get("risk_state", "neutral")
        trend = regime.get("btc_trend", "chop")
        if risk == "risk_on":
            narrative["sentiment"] = "bullish" if trend == "up" else "risk_on"
        elif risk == "risk_off":
            narrative["sentiment"] = "bearish" if trend == "down" else "risk_off"
        else:
            narrative["sentiment"] = "neutral"

        # ── Breadth strength ──
        m = regime.get("metrics", {})
        pump_pct = m.get("breadth_pump10_pct", 0) or 0
        dump_pct = m.get("breadth_dump10_pct", 0) or 0
        if pump_pct >= 0.20:
            narrative["breadth_strength"] = "strong_bullish"
        elif dump_pct >= 0.20:
            narrative["breadth_strength"] = "strong_bearish"
        elif pump_pct >= 0.10:
            narrative["breadth_strength"] = "moderate_bullish"
        elif dump_pct >= 0.10:
            narrative["breadth_strength"] = "moderate_bearish"
        else:
            narrative["breadth_strength"] = "narrow"

        # ── Volatility context ──
        vol = regime.get("vol_state", "contraction")
        atr_ratio = m.get("atr_ratio", 1.0) or 1.0
        if vol == "expansion":
            narrative["volatility_context"] = "elevated" if atr_ratio >= 1.5 else "expanding"
        else:
            narrative["volatility_context"] = "compressed" if atr_ratio <= 0.7 else "stable"
    else:
        narrative["sentiment"] = "unknown"
        narrative["breadth_strength"] = "unknown"
        narrative["volatility_context"] = "unknown"

    # ── Geo driver ──
    if geo.get("active"):
        top_events = geo.get("top", [])
        if top_events:
            top_ev = top_events[0]
            headline = top_ev.get("headline", "")
            # Extract a short driver label from the headline
            narrative["geo_driver"] = _extract_geo_theme(headline, top_ev.get("severity", 0))

    # ── Build summary sentence ──
    parts = []
    if regime:
        parts.append(f"{regime['regime_code'].replace('_', ' ').title()}")
    else:
        parts.append("Regime unknown")

    if narrative["market_driver"] != "quiet":
        parts.append(f"led by {narrative['market_driver']}")

    parts.append(f"sentiment {narrative['sentiment']}")

    if narrative["geo_driver"]:
        parts.append(f"geo: {narrative['geo_driver']}")

    narrative["summary"] = " | ".join(parts)

    return narrative


def _extract_geo_theme(headline: str, severity: int) -> str:
    """Extract a short geo theme from a headline."""
    hl = headline.lower()
    themes = [
        ("iran", "Iran tensions"),
        ("china", "China tensions"),
        ("russia", "Russia tensions"),
        ("ukraine", "Ukraine conflict"),
        ("tariff", "Trade tensions"),
        ("sanction", "Sanctions"),
        ("middle east", "Middle East conflict"),
        ("israel", "Israel conflict"),
        ("nuclear", "Nuclear threat"),
        ("fed ", "Fed policy"),
        ("rate", "Rate policy"),
        ("election", "Elections"),
        ("military", "Military action"),
        ("war", "Military conflict"),
    ]
    for keyword, theme in themes:
        if keyword in hl:
            return theme
    return f"Sev-{severity} event" if severity >= 80 else "geopolitical"


# ── Weights ──────────────────────────────────────────────────

def fetch_weights(pg_dsn: str) -> dict:
    """Fetch current adaptive weights status for state snapshot."""
    from hltrader.config import settings
    result = {
        "enabled": settings.HL_ADAPTIVE_WEIGHTS_ENABLED,
        "source": "STATIC",
        "asof_date": None,
        "sample_size": 0,
        "values": {
            "pump": settings.HL_SCAN_W_PUMP,
            "oi": settings.HL_SCAN_W_OI,
            "funding": settings.HL_SCAN_W_FUNDING,
            "accel": settings.HL_SCAN_W_ACCEL,
        },
    }
    if not settings.HL_ADAPTIVE_WEIGHTS_ENABLED:
        return result
    try:
        from hltrader.analysis.adaptive_weights import get_weights_for_regime
        sel = get_weights_for_regime(
            pg_dsn,
            static_weights={
                "w_pump": settings.HL_SCAN_W_PUMP,
                "w_oi": settings.HL_SCAN_W_OI,
                "w_funding": settings.HL_SCAN_W_FUNDING,
                "w_accel": settings.HL_SCAN_W_ACCEL,
            },
        )
        result["source"] = sel.source
        result["asof_date"] = str(sel.asof_date) if sel.asof_date else None
        result["sample_size"] = sel.sample_size
        result["values"] = {
            "pump": round(sel.weights["w_pump"], 1),
            "oi": round(sel.weights["w_oi"], 1),
            "funding": round(sel.weights["w_funding"], 1),
            "accel": round(sel.weights["w_accel"], 1),
        }
    except Exception as exc:
        log.warning("fetch_weights failed: %s", exc)
    return result


# ── Build ────────────────────────────────────────────────────

def fetch_token_personalities(pg_dsn: str) -> list[dict]:
    """Fetch top token personalities for state snapshot."""
    result = []
    try:
        from hltrader.analysis.token_memory import clear_cache, _load_all_personalities, _memory_cache
        clear_cache()
        _load_all_personalities(pg_dsn)
        ranked = sorted(_memory_cache.values(),
                        key=lambda tp: tp.confidence_score, reverse=True)
        for tp in ranked[:5]:
            result.append({
                "symbol": tp.symbol,
                "label": tp.personality_label,
                "confidence": round(tp.confidence_score, 0),
                "win_24h": round(tp.win_24h, 2) if tp.win_24h is not None else None,
                "avg_ret_24h": round(tp.avg_ret_24h, 2) if tp.avg_ret_24h is not None else None,
                "trend_follow": round(tp.trend_follow_score, 0),
                "mean_reversion": round(tp.mean_reversion_score, 0),
                "best_regime": tp.best_regime,
            })
    except Exception as exc:
        log.warning("fetch_token_personalities failed: %s", exc)
    return result


def build_state(pg_dsn: str) -> dict:
    """Build complete market state snapshot."""
    now = datetime.now(timezone.utc)
    errors = []

    regime = fetch_regime(pg_dsn)
    if regime is None:
        errors.append("regime: no recent snapshot (>60m old or DB error)")

    alerts = fetch_alerts(pg_dsn)
    performance = fetch_performance(pg_dsn)
    geo = fetch_geo(pg_dsn)
    system = fetch_system()

    narrative = build_narrative(regime, alerts, geo)

    weights = fetch_weights(pg_dsn)

    token_personalities = fetch_token_personalities(pg_dsn)

    state = {
        "ts": now.isoformat(),
        "regime": regime,
        "narrative": narrative,
        "alerts": alerts,
        "performance": performance,
        "geo": geo,
        "system": system,
        "weights": weights,
        "token_personalities": token_personalities,
    }

    if errors:
        state["errors"] = errors

    return state


def _format_md(state: dict) -> str:
    """Format state as human-readable markdown."""
    lines = [f"# Market State — {state['ts'][:19]}Z", ""]

    # Regime
    r = state.get("regime")
    if r:
        lines.append(f"## Regime: {r['regime_code']}")
        lines.append(f"- Risk: **{r['risk_state']}** | BTC: {r['btc_trend']} | Vol: {r['vol_state']}")
        m = r.get("metrics", {})
        if m.get("btc_slope_4h") is not None:
            lines.append(f"- Slope: {m['btc_slope_4h']:.4f} | ATR ratio: {m.get('atr_ratio', 0):.2f}")
        lines.append("")
    else:
        lines.append("## Regime: UNAVAILABLE")
        lines.append("")

    # Narrative
    n = state.get("narrative", {})
    if n and n.get("summary"):
        lines.append("## Market Narrative")
        lines.append(f"- {n['summary']}")
        lines.append(f"- Driver: **{n.get('market_driver', '?')}** | Sentiment: {n.get('sentiment', '?')}")
        lines.append(f"- Breadth: {n.get('breadth_strength', '?')} | Volatility: {n.get('volatility_context', '?')}")
        heat = n.get("sector_heat", [])
        if heat:
            heat_str = ", ".join(f"{h['label']} ({h['count']})" for h in heat)
            lines.append(f"- Sector heat: {heat_str}")
        geo_d = n.get("geo_driver")
        if geo_d:
            lines.append(f"- Geo driver: {geo_d}")
        lines.append("")

    # Alerts
    a = state.get("alerts", {})
    counts = a.get("last_24h_counts", {})
    total = sum(counts.values())
    lines.append(f"## Alerts (24h): {total} total")
    for k, v in counts.items():
        if v > 0:
            lines.append(f"- {k}: {v}")
    shadow = a.get("shadow_blocked_24h", 0)
    if shadow:
        lines.append(f"- Shadow blocked: {shadow}")
    lines.append("")

    top = a.get("top_active", [])
    if top:
        lines.append("### Top Signals")
        for t in top:
            conv = f" | Conv: {t['conviction']:.0f}" if t.get("conviction") else ""
            lines.append(f"- **{t['symbol']}** ({t['lane']}) Score: {t['score']:.0f}{conv} Tier: {t.get('tier', '-')}")
        lines.append("")

    # Performance
    p = state.get("performance", {})
    tiers = p.get("tiers", {})
    if tiers:
        lines.append(f"## Performance ({p.get('window_days', 7)}d)")
        for tier, d in tiers.items():
            w24 = f"{d['win_24h']:.0%}" if d.get("win_24h") is not None else "n/a"
            avg = f"{d['avg_24h']:+.1f}%" if d.get("avg_24h") is not None else "n/a"
            lines.append(f"- Tier {tier}: {d['count']} alerts | Win24h: {w24} | Avg24h: {avg}")
        lines.append("")

    by_regime = p.get("by_regime", [])
    if by_regime:
        lines.append("### By Regime")
        for br in by_regime:
            w24 = f"{br['win_24h']:.0%}" if br.get("win_24h") is not None else "n/a"
            avg = f"{br['avg_24h']:+.1f}%" if br.get("avg_24h") is not None else "n/a"
            lines.append(f"- {br['regime_code']}: {br['count']} alerts | Win24h: {w24} | Avg24h: {avg}")
        lines.append("")

    # Weights
    wt = state.get("weights", {})
    if wt:
        src_label = wt.get("source", "STATIC")
        asof = wt.get("asof_date") or "n/a"
        n = wt.get("sample_size", 0)
        v = wt.get("values", {})
        lines.append(f"## Weights: {src_label} (asof {asof}, n={n})")
        lines.append(f"- P={v.get('pump', 0):.1f} / O={v.get('oi', 0):.1f}"
                     f" / F={v.get('funding', 0):.1f} / A={v.get('accel', 0):.1f}")
        lines.append("")

    # Token Personalities
    tp_list = state.get("token_personalities", [])
    if tp_list:
        lines.append("## Token Personalities")
        for tp in tp_list:
            w24 = f"Win24h: {tp['win_24h']:.0%}" if tp.get("win_24h") is not None else ""
            avg24 = f"Avg24h: {tp['avg_ret_24h']:+.1f}%" if tp.get("avg_ret_24h") is not None else ""
            regime_str = f"Best: {tp['best_regime']}" if tp.get("best_regime") else ""
            parts = [x for x in [w24, avg24, regime_str] if x]
            detail = f" | {' | '.join(parts)}" if parts else ""
            lines.append(f"- **{tp['symbol']}**: {tp['label']} (conf {tp['confidence']:.0f}){detail}")
        lines.append("")

    # Geo
    g = state.get("geo", {})
    if g.get("active"):
        lines.append("## Geo Events (active)")
        for ev in g.get("top", []):
            lines.append(f"- [{ev['severity']}] {ev['headline']}")
        lines.append("")
    else:
        lines.append("## Geo: quiet")
        lines.append("")

    # System
    s = state.get("system", {}).get("services", {})
    if s:
        lines.append("## Services")
        for name, status in s.items():
            icon = "OK" if status == "active" else "DOWN"
            lines.append(f"- {name}: {icon}")
        lines.append("")

    # Errors
    errs = state.get("errors", [])
    if errs:
        lines.append("## Warnings")
        for e in errs:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)


def write_state(state: dict) -> tuple[Path, Path]:
    """Write state JSON + MD atomically. Returns (json_path, md_path)."""
    _ensure_dir()
    _atomic_write(STATE_JSON, json.dumps(state, indent=2, default=str))
    _atomic_write(STATE_MD, _format_md(state))
    return STATE_JSON, STATE_MD
