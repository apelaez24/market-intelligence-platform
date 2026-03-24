"""Safety validation for stop-loss orders, size/price rounding, slippage."""

from __future__ import annotations

from hltrader.client import get_info


def validate_stop_loss(
    entry_px: float,
    trigger_px: float,
    is_long: bool,
    *,
    force: bool = False,
) -> None:
    """Raise ValueError if the SL is on the wrong side of entry.

    For longs: SL must be *below* entry price.
    For shorts: SL must be *above* entry price.
    Pass ``force=True`` to override.
    """
    if force:
        return
    if is_long and trigger_px >= entry_px:
        raise ValueError(
            f"Stop-loss {trigger_px} must be below entry {entry_px} for a LONG. "
            "Use --force to override."
        )
    if not is_long and trigger_px <= entry_px:
        raise ValueError(
            f"Stop-loss {trigger_px} must be above entry {entry_px} for a SHORT. "
            "Use --force to override."
        )


def get_sz_decimals(coin: str) -> int:
    """Return the size-decimal count for *coin* from exchange metadata."""
    info = get_info()
    meta = info.meta()
    for item in meta["universe"]:
        if item["name"] == coin:
            return item["szDecimals"]
    raise ValueError(f"Coin {coin!r} not found in exchange metadata")


def round_size(sz: float, coin: str) -> float:
    """Round *sz* to the exchange-allowed number of decimals for *coin*."""
    decimals = get_sz_decimals(coin)
    return round(sz, decimals)


def round_price(px: float) -> float:
    """Round price to 5 significant figures and 6 decimal places (perp rules)."""
    return round(float(f"{px:.5g}"), 6)


def compute_slippage_price(
    px: float,
    is_buy: bool,
    slippage: float = 0.05,
) -> float:
    """Compute the limit price with slippage applied.

    Replicates the SDK's ``Exchange._slippage_price`` logic for trigger orders.
    """
    adjusted = px * (1 + slippage) if is_buy else px * (1 - slippage)
    return round(float(f"{adjusted:.5g}"), 6)


def sl_trigger_px_from_pct(entry_px: float, pct: float, is_long: bool) -> float:
    """Compute an absolute SL trigger price from a percentage distance.

    *pct* is expressed as a positive number (e.g. 2.5 means 2.5%).
    """
    frac = pct / 100
    if is_long:
        return round_price(entry_px * (1 - frac))
    else:
        return round_price(entry_px * (1 + frac))


def tp_trigger_px_from_pct(entry_px: float, pct: float, is_long: bool) -> float:
    """Compute an absolute TP trigger price from a percentage distance."""
    frac = pct / 100
    if is_long:
        return round_price(entry_px * (1 + frac))
    else:
        return round_price(entry_px * (1 - frac))
