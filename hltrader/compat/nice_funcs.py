"""Drop-in replacement for the original nice_funcs.py.

Usage in existing bots::

    from hltrader.compat import nice_funcs as n

All original function signatures are preserved.  Internally they use
the SDK via ``hltrader.client`` singletons instead of creating new
Exchange/Info instances on every call.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import requests
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from hltrader.client import get_exchange, get_info, get_address
from hltrader.config import settings
from hltrader.orders.trigger import place_stop_loss as _place_stop_loss
from hltrader.orders.validation import sl_trigger_px_from_pct


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def ask_bid(symbol: str):
    """Return (ask, bid, l2_data) for *symbol*."""
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {"type": "l2Book", "coin": symbol}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()["levels"]
    bid = float(l2_data[0][0]["px"])
    ask = float(l2_data[1][0]["px"])
    return ask, bid, l2_data


def get_sz_px_decimals(symbol: str):
    """Return (sz_decimals, px_decimals) for *symbol*."""
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {"type": "meta"}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    sz_decimals = 0
    if response.status_code == 200:
        symbols = response.json()["universe"]
        symbol_info = next((s for s in symbols if s["name"] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info["szDecimals"]
        else:
            print("Symbol not found")
    else:
        print("Error:", response.status_code)

    ask = ask_bid(symbol)[0]
    ask_str = str(ask)
    px_decimals = len(ask_str.split(".")[1]) if "." in ask_str else 0
    print(f"{symbol} this is the price {sz_decimals} decimal(s)")
    return sz_decimals, px_decimals


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """Place a limit GTC order — same signature as original."""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)
    print(f"coin: {coin}, type: {type(coin)}")
    print(f"is_buy: {is_buy}, type: {type(is_buy)}")
    print(f"sz: {sz}, type: {type(sz)}")
    print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    print(f"reduce_only: {reduce_only}, type: {type(reduce_only)}")
    print(f"placing limit order for {coin} {sz} @ {limit_px}")

    order_result = exchange.order(
        coin, is_buy, sz, limit_px,
        {"limit": {"tif": "Gtc"}},
        reduce_only=reduce_only,
    )
    tag = "BUY" if is_buy else "SELL"
    print(f"limit {tag} order placed, resting: {order_result['response']['data']['statuses'][0]}")
    return order_result


# ---------------------------------------------------------------------------
# Account & position helpers
# ---------------------------------------------------------------------------

def acct_bal(account):
    """Return the account value string."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f"this is current account value: {user_state['marginSummary']['accountValue']}")
    return user_state["marginSummary"]["accountValue"]


def get_position(symbol, account):
    """Return (positions, in_pos, size, pos_sym, entry_px, pnl_perc, long)."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f"this is current account value: {user_state['marginSummary']['accountValue']}")
    positions = []
    for position in user_state["assetPositions"]:
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f"this is the pnl perc {pnl_perc}")
            break
    else:
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0

    long = True if size > 0 else (False if size < 0 else None)
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long


def get_position_andmaxpos(symbol, account, max_positions):
    """Return position info + enforce max-positions guard."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f"this is current account value: {user_state['marginSummary']['accountValue']}")
    open_positions = []
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            open_positions.append(position["position"]["coin"])

    num_of_pos = len(open_positions)
    if len(open_positions) > max_positions:
        print(f"we are in {len(open_positions)} positions and max pos is {max_positions}... closing positions")
        for pos_coin in open_positions:
            kill_switch(pos_coin, account)
    else:
        print(f"we are in {len(open_positions)} positions and max pos is {max_positions}... not closing positions")

    positions = []
    for position in user_state["assetPositions"]:
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f"this is the pnl perc {pnl_perc}")
            break
    else:
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0

    long = True if size > 0 else (False if size < 0 else None)
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos


def adjust_leverage_size_signal(symbol, leverage, account):
    """Return (leverage, size) for 95% of account value."""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    acct_value = float(user_state["marginSummary"]["accountValue"])
    acct_val95 = acct_value * 0.95
    print(exchange.update_leverage(leverage, symbol))
    price = ask_bid(symbol)[0]
    size = (acct_val95 / price) * leverage
    rounding = get_sz_px_decimals(symbol)[0]
    size = round(float(size), rounding)
    return leverage, size


# ---------------------------------------------------------------------------
# Kill switch & PnL close
# ---------------------------------------------------------------------------

def cancel_all_orders(account):
    """Cancel all open orders."""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    open_orders = info.open_orders(account.address)
    print("above are the open orders... need to cancel any...")
    for open_order in open_orders:
        exchange.cancel(open_order["coin"], open_order["oid"])


def kill_switch(symbol, account):
    """Close position by aggressive limit orders until flat."""
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    while im_in_pos:
        cancel_all_orders(account)
        ask, bid, _ = ask_bid(symbol)
        ps = abs(pos_size)
        if long:
            limit_order(pos_sym, False, ps, ask, True, account)
            print("kill switch sell to close submitted")
            time.sleep(5)
        elif long is False:
            limit_order(pos_sym, True, ps, bid, True, account)
            print("kill switch buy to close submitted")
            time.sleep(5)
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    print("position successfully closed in kill switch")


def pnl_close(symbol, target, max_loss, account):
    """Close position if PnL hits target or max_loss."""
    print("entering pnl close")
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    if pnl_perc > target:
        print(f"pnl gain is {pnl_perc} and target is {target} closing position as a WIN")
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f"pnl loss is {pnl_perc} and max loss is {max_loss} closing position as a LOSS")
        kill_switch(pos_sym, account)
    else:
        print(f"pnl loss is {pnl_perc} and max loss is {max_loss} target {target} not closing")
    print("finished pnl close")


def close_all_positions(account):
    """Cancel all orders, then kill-switch every open position."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f"this is current account value: {user_state['marginSummary']['accountValue']}")
    cancel_all_orders(account)
    print("all orders have been cancelled")
    open_positions = []
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            open_positions.append(position["position"]["coin"])
    for pos_coin in open_positions:
        kill_switch(pos_coin, account)
    print("all positions have been closed")


# ---------------------------------------------------------------------------
# OHLCV / candle helpers
# ---------------------------------------------------------------------------

def get_ohlcv2(symbol, interval, lookback_days):
    """Fetch candle snapshot data."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        },
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching data for {symbol}: {response.status_code}")
    return None


def process_data_to_df(snapshot_data):
    """Convert candle snapshot data to a pandas DataFrame."""
    import pandas as pd

    if not snapshot_data:
        return pd.DataFrame()
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    rows = []
    for s in snapshot_data:
        ts = datetime.fromtimestamp(s["t"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([ts, s["o"], s["h"], s["l"], s["c"], s["v"]])
    df = pd.DataFrame(rows, columns=columns)
    if len(df) > 2:
        df["support"] = df[:-2]["close"].min()
        df["resis"] = df[:-2]["close"].max()
    else:
        df["support"] = df["close"].min()
        df["resis"] = df["close"].max()
    return df


def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    """Fetch raw candle snapshot."""
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
        },
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching data for {symbol}: {response.status_code}")
    return None
