"""Cached SDK wrapper — Exchange + Info singletons, with dry-run support."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from rich.console import Console

from hltrader.config import settings

_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Dry-run exchange: logs every write call instead of sending it
# ---------------------------------------------------------------------------

class DryRunExchange:
    """Drop-in replacement for ``Exchange`` that prints what *would* happen.

    Read operations (via the inner ``info`` attribute) still hit the real API
    so you see real account data.  All write operations (order, cancel,
    market_open, market_close, update_leverage, …) are intercepted and logged.
    """

    def __init__(self, real_exchange: Exchange) -> None:
        self._real = real_exchange
        # Expose info so SDK internals that read metadata still work
        self.info = real_exchange.info

    # -- helpers used internally by the SDK --
    def _slippage_price(self, *args: Any, **kwargs: Any) -> float:
        return self._real._slippage_price(*args, **kwargs)

    # -- write operations: log instead of execute --

    def order(self, name: str, is_buy: bool, sz: float, limit_px: float,
              order_type: Any, reduce_only: bool = False, **kw: Any) -> Any:
        side = "BUY" if is_buy else "SELL"
        ro = " reduce_only" if reduce_only else ""
        _console.print(
            f"[bold magenta][DRY-RUN][/bold magenta] order: {side} {sz} {name} "
            f"@ {limit_px}  type={order_type}{ro}"
        )
        return _dry_run_result(name, is_buy, sz, limit_px)

    def bulk_orders(self, order_requests: list, builder: Any = None) -> Any:
        for req in order_requests:
            self.order(
                req["coin"], req["is_buy"], req["sz"], req["limit_px"],
                req["order_type"], req.get("reduce_only", False),
            )
        return _dry_run_result("bulk", True, 0, 0)

    def market_open(self, name: str, is_buy: bool, sz: float, **kw: Any) -> Any:
        side = "LONG" if is_buy else "SHORT"
        _console.print(
            f"[bold magenta][DRY-RUN][/bold magenta] market_open: {side} {sz} {name}"
        )
        return _dry_run_result(name, is_buy, sz, 0)

    def market_close(self, coin: str, sz: Optional[float] = None, **kw: Any) -> Any:
        _console.print(
            f"[bold magenta][DRY-RUN][/bold magenta] market_close: {coin}"
            + (f" sz={sz}" if sz else " (full)")
        )
        return _dry_run_result(coin, True, sz or 0, 0)

    def cancel(self, name: str, oid: int) -> Any:
        _console.print(
            f"[bold magenta][DRY-RUN][/bold magenta] cancel: {name} oid={oid}"
        )
        return {"status": "ok", "response": {"type": "cancel", "data": {"statuses": ["dry-run"]}}}

    def bulk_cancel(self, cancel_requests: list) -> Any:
        for req in cancel_requests:
            self.cancel(req["coin"], req["oid"])
        return {"status": "ok"}

    def update_leverage(self, leverage: int, name: str, is_cross: bool = True) -> Any:
        _console.print(
            f"[bold magenta][DRY-RUN][/bold magenta] update_leverage: "
            f"{name} → {leverage}x ({'cross' if is_cross else 'isolated'})"
        )
        return {"status": "ok"}

    def modify_order(self, *args: Any, **kwargs: Any) -> Any:
        _console.print(f"[bold magenta][DRY-RUN][/bold magenta] modify_order: {args}")
        return _dry_run_result("modify", True, 0, 0)

    # Forward attribute access for anything else to the real exchange
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _dry_run_result(coin: str, is_buy: bool, sz: float, px: float) -> dict:
    """Return a fake order-result dict that matches the real SDK shape."""
    return {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {
                        "resting": {
                            "oid": 0,
                            "coin": coin,
                            "side": "B" if is_buy else "A",
                            "sz": str(sz),
                            "limitPx": str(px),
                        },
                        "dry_run": True,
                    }
                ]
            },
        },
    }


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_account() -> LocalAccount:
    key = settings.HL_PRIVATE_KEY
    if not key:
        raise RuntimeError(
            "HL_PRIVATE_KEY is not set. "
            "Export it as an env var or put it in a .env file."
        )
    return eth_account.Account.from_key(key)


@lru_cache(maxsize=1)
def get_info() -> Info:
    return Info(settings.effective_api_url, skip_ws=True)


@lru_cache(maxsize=1)
def _get_real_exchange() -> Exchange:
    return Exchange(get_account(), settings.effective_api_url)


def get_exchange() -> Exchange | DryRunExchange:
    """Return a ``DryRunExchange`` when dry-run mode is active."""
    real = _get_real_exchange()
    if settings.HL_DRY_RUN:
        return DryRunExchange(real)
    return real


def get_address() -> str:
    return get_account().address
