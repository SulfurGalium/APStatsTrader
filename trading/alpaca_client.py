from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderStatus
from alpaca.trading.requests import MarketOrderRequest

from datetime import datetime, timedelta, timezone
from alpaca.data.enums import DataFeed

from config import API_KEY, SECRET_KEY, FRED_API_KEY, LIVE_BAR_TIMEFRAME_MIN, MACRO_CACHE_CSV
from data.fred_cache import get_latest_treasury_yield


trading_client: TradingClient | None = None
stock_data_client: StockHistoricalDataClient | None = None


def _require_credentials() -> tuple[str, str]:
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY before paper trading."
        )
    return API_KEY, SECRET_KEY


def get_trading_client() -> TradingClient:
    global trading_client
    if trading_client is None:
        api_key, secret_key = _require_credentials()
        trading_client = TradingClient(api_key, secret_key, paper=True)
    return trading_client


def get_stock_data_client() -> StockHistoricalDataClient:
    global stock_data_client
    if stock_data_client is None:
        api_key, secret_key = _require_credentials()
        stock_data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return stock_data_client


def get_market_clock():
    """
    Return Alpaca's U.S. equity market clock.
    """
    return get_trading_client().get_clock()


def is_us_equity_market_open() -> tuple[bool, Optional[datetime]]:
    """
    Check whether regular U.S. equity trading is currently open.

    SPY trades on the U.S. equity market, so Alpaca's market clock is the
    source of truth for whether the live loop should process bars and orders.
    """
    clock = get_market_clock()
    next_open = getattr(clock, "next_open", None)
    return bool(getattr(clock, "is_open", False)), next_open


def wait_for_fill(order_id: str, timeout: int = 30) -> float:
    """
    Wait until order is filled and return filled quantity (paper account).
    """
    start = time.time()
    while time.time() - start < timeout:
        order = get_trading_client().get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return float(order.filled_qty)
        time.sleep(1.0)
    raise TimeoutError("Order did not fill in time")


def get_latest_stock_price(symbol: str) -> Optional[float]:
    """
    Uses the Latest Bar endpoint, which is usually REAL-TIME 
    on IEX even for free users.
    """
    from alpaca.data.requests import StockLatestBarRequest
    
    request = StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
    res = get_stock_data_client().get_stock_latest_bar(request)
    
    if symbol in res:
        return float(res[symbol].close)
    return None


def submit_stock_bracket_order(
    symbol: str,
    side: str,
    qty: int,
    target_price: float,
    stop_price: float,
):
    """
    Submit a paper-trading stock bracket order via Alpaca TradingClient.
    """
    try:
        order = get_trading_client().submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(float(target_price), 2)},
                stop_loss={"stop_price": round(float(stop_price), 2)},
            )
        )
        print(
            f"[ORDER] {side.upper()} {symbol} qty={qty} "
            f"tp={target_price:.2f} sl={stop_price:.2f}"
        )
        return order
    except Exception as exc:
        print(f"Error placing stock bracket order: {exc}")
        return None


def get_account_equity() -> Optional[float]:
    """
    Return current account equity when available.
    """
    try:
        account = get_trading_client().get_account()
        return float(account.equity)
    except Exception as exc:
        print(f"Could not fetch account equity: {exc}")
        return None


def has_open_position(symbol: str) -> bool:
    """
    Check whether a symbol currently has an open position.
    """
    try:
        positions = get_trading_client().get_all_positions()
        return any(getattr(position, "symbol", None) == symbol for position in positions)
    except Exception as exc:
        print(f"Could not check open positions: {exc}")
        return False


def close_stock_position(symbol: str):
    """
    Close an open position for a symbol if one exists.
    """
    try:
        result = get_trading_client().close_position(symbol)
        print(f"[CLOSE] Requested close for {symbol}")
        return result
    except Exception as exc:
        print(f"Could not close position for {symbol}: {exc}")
        return None


def get_last_60_stock_bars(symbol: str, lookback_bars: int = 61) -> pd.DataFrame:
    """
    Fetches the most recent stock data by requesting a window and slicing the end.
    """
    client = get_stock_data_client()
    
    # Define a wide window to ensure we cover weekends and the overnight gap
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=3)  # Go back 3 days to safely handle weekends

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(LIVE_BAR_TIMEFRAME_MIN, TimeFrameUnit.Minute),
        start=start_time,
        end=now,
        feed=DataFeed.IEX,  # Use IEX for Free Tier
        # We REMOVE the limit parameter here so we don't get stuck in the past
    )
    
    try:
        bars = client.get_stock_bars(request_params)
        df = bars.df

        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol}. Market might be closed.")

        # Handle MultiIndex if necessary
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        # CRITICAL: We take the LAST 'lookback_bars' to get the current data
        df = df.tail(lookback_bars)

        # Standardize timezone and formatting
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        df = df.drop(columns=["trade_count", "vwap"], errors="ignore")
        df.columns = [col.capitalize() for col in df.columns]
        
        # Add Macro data
        latest_yield = get_latest_treasury_yield(
            at_time=df.index.max(),
            api_key=FRED_API_KEY,
            cache_name=MACRO_CACHE_CSV,
        )
        df["UST10Y"] = latest_yield
        df.index.name = "Time"
        
        return df

    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        raise
