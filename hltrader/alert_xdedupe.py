"""Cross-alert de-duplication helpers.

Shared logic — duplicated in hltrader and ibkr_agent because they run
in separate venvs.  Keep both copies in sync.

Fail-open design: if the DB is down, alerts still send.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)


def record_alert_key(
    conn,
    *,
    symbol: str,
    direction: str,
    alert_type: str,
    source: str,
) -> bool:
    """Record (upsert) an alert key in alert_cross_dedupe.

    Returns True on success. Caller provides a connection with
    autocommit=False; this function commits on success.
    """
    key = f"{symbol}|{direction}|{alert_type}"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO alert_cross_dedupe (key, symbol, direction, source)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (key) DO UPDATE SET last_seen = now()""",
                (key, symbol, direction, source),
            )
        conn.commit()
        log.debug("xdedupe: recorded %s (source=%s)", key, source)
        return True
    except Exception:
        log.exception("xdedupe: failed to record %s", key)
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def check_recent_alert(
    conn,
    *,
    symbol: str,
    direction: str,
    alert_type: str,
    window_minutes: int = 30,
) -> bool:
    """Check if an alert key was recorded within the lookback window.

    Returns True if a matching row exists with last_seen within window.
    Fail-open: returns False on DB error (allow the alert).
    """
    key = f"{symbol}|{direction}|{alert_type}"
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT 1 FROM alert_cross_dedupe
                   WHERE key = %s AND last_seen >= %s""",
                (key, cutoff),
            )
            found = cur.fetchone() is not None
        log.debug("xdedupe: check %s within %dm -> %s", key, window_minutes, found)
        return found
    except Exception:
        log.exception("xdedupe: failed to check %s", key)
        return False
