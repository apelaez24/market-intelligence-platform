"""Data models for positions, coin specs, stop configs, etc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    coin: str
    size: float          # signed: positive = long, negative = short
    entry_px: float
    unrealized_pnl: float
    return_on_equity: float
    leverage_type: str   # "cross" | "isolated"
    leverage_value: int
    liquidation_px: Optional[float]
    margin_used: float
    position_value: float

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def abs_size(self) -> float:
        return abs(self.size)

    @classmethod
    def from_user_state(cls, item: dict) -> Position:
        pos = item["position"]
        lev = pos["leverage"]
        return cls(
            coin=pos["coin"],
            size=float(pos["szi"]),
            entry_px=float(pos["entryPx"]) if pos.get("entryPx") else 0.0,
            unrealized_pnl=float(pos["unrealizedPnl"]),
            return_on_equity=float(pos["returnOnEquity"]),
            leverage_type=lev["type"],
            leverage_value=int(lev["value"]),
            liquidation_px=float(pos["liquidationPx"]) if pos.get("liquidationPx") else None,
            margin_used=float(pos["marginUsed"]),
            position_value=float(pos["positionValue"]),
        )


@dataclass
class CoinSpec:
    name: str
    sz_decimals: int
    max_leverage: Optional[int] = None

    @classmethod
    def from_meta(cls, item: dict) -> CoinSpec:
        return cls(
            name=item["name"],
            sz_decimals=item["szDecimals"],
            max_leverage=item.get("maxLeverage"),
        )


@dataclass
class TriggerOrderInfo:
    oid: int
    coin: str
    side: str            # "A" (ask/sell) or "B" (bid/buy)
    size: float
    limit_px: float
    trigger_px: float
    trigger_condition: str
    is_trigger: bool
    reduce_only: bool
    order_type: str
    is_position_tpsl: bool

    @classmethod
    def from_frontend_order(cls, item: dict) -> TriggerOrderInfo:
        return cls(
            oid=item["oid"],
            coin=item["coin"],
            side=item["side"],
            size=float(item["sz"]),
            limit_px=float(item["limitPx"]),
            trigger_px=float(item.get("triggerPx", 0)),
            trigger_condition=item.get("triggerCondition", ""),
            is_trigger=item.get("isTrigger", False),
            reduce_only=item.get("reduceOnly", False),
            order_type=item.get("orderType", ""),
            is_position_tpsl=item.get("isPositionTpsl", False),
        )


@dataclass
class StopConfig:
    coin: str
    trigger_px: float
    is_long: bool        # True = SL below entry, False = SL above entry
    size: float          # size to close


@dataclass
class BracketConfig:
    coin: str
    is_long: bool
    size: float
    leverage: int
    sl_px: Optional[float] = None
    sl_pct: Optional[float] = None
    tp_px: Optional[float] = None
    tp_pct: Optional[float] = None
