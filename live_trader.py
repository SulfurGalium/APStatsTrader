from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Assuming these are your local modules
from config import (
    SYMBOL,
    CONTEXT_SIZE,
    DEFAULT_EMA_SPAN,
    DEFAULT_MOMENTUM_WEIGHT,
    DEFAULT_RISK_PCT,
    LIVE_BAR_TIMEFRAME_MIN,
    MODEL_PREFIX,
)

from trading.alpaca_client import (
    close_stock_position,
    get_account_equity,
    get_last_60_stock_bars, # Ensure this uses the 'end' parameter as shown below
    has_open_position,
    is_us_equity_market_open,
    submit_stock_bracket_order,
)
from trading.signal_engine import (
    build_processed_features,
    evaluate_trade_plan,
    sample_return_distribution,
)
from training.io import load_model_and_scaler


def build_live_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.Index, pd.DataFrame]:
    features, aligned_raw = build_processed_features(df, feature_cols)
    return features, aligned_raw.index, aligned_raw


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seconds_until_next_open(next_open) -> int:
    if next_open is None:
        return 300

    next_open_ts = pd.Timestamp(next_open)
    if next_open_ts.tzinfo is None:
        next_open_ts = next_open_ts.tz_localize("UTC")
    else:
        next_open_ts = next_open_ts.tz_convert("UTC")

    now_utc = pd.Timestamp.now(tz="UTC")
    return max(30, int((next_open_ts - now_utc).total_seconds()))


def decide_trade_from_model(
    model,
    scaler,
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    context_size: int,
    risk_pct: float,
    equity: float,
    ema_span: int,
    momentum_weight: float,
) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[float], str]:
    features, _, aligned_raw = build_live_features(df_raw, feature_cols)
    
    if len(features) < context_size:
        reason = f"data padding: {len(features)}/{context_size} processed rows"
        print(f"[NO SIGNAL] {reason}")
        return None, None, None, None, reason

    scaled = scaler.transform(features)
    context = scaled[-context_size:]
    feature_dim = scaled.shape[1]
    
    mu, sigma = sample_return_distribution(
        model=model,
        context=context,
        scaler=scaler,
        feature_dim=feature_dim,
    )

    current_price = float(aligned_raw["Close"].iloc[-1])
    ema_val = float(aligned_raw["Close"].ewm(span=ema_span, adjust=False).mean().iloc[-1])
    
    trade_plan, rejection_reason = evaluate_trade_plan(
        mu=mu,
        sigma=sigma,
        current_price=current_price,
        recent_closes=aligned_raw["Close"].iloc[-context_size:],
        ema_val=ema_val,
        equity=equity,
        risk_pct=risk_pct,
        momentum_weight=momentum_weight,
    )

    print(
        f"[MODEL] price={current_price:.2f} ema={ema_val:.2f} "
        f"mu={mu:.6f} sigma={sigma:.6f}"
    )

    if trade_plan is None:
        print(f"[NO SIGNAL] {rejection_reason}")
        return None, None, None, None, rejection_reason

    print(
        f"[SIGNAL] {trade_plan.side} | Qty: {trade_plan.qty} | "
        f"Entry: {trade_plan.current_price:.2f} | TP: {trade_plan.target_price:.2f} | "
        f"SL: {trade_plan.stop_price:.2f}"
    )
    return trade_plan.side, trade_plan.qty, trade_plan.target_price, trade_plan.stop_price, "accepted"


def live_paper_loop(
    model_prefix: str = MODEL_PREFIX,
    feature_cols: List[str] | None = None,
    poll_seconds: int | None = None,
    starting_equity: float = 10_000.0,
    use_cosine_beta: bool = False,
    device_name: str = "auto",
):
    device = resolve_device(device_name)
    model, scaler = load_model_and_scaler(model_prefix, device=device, use_cosine_beta=use_cosine_beta)
    
    if model is None or scaler is None:
        raise RuntimeError("Model/Scaler load failed.")

    feature_cols = feature_cols or ["Open", "High", "Low", "Close", "Volume", "UST10Y"]
    poll_seconds = poll_seconds or LIVE_BAR_TIMEFRAME_MIN * 60
    
    # Timing variables
    bar_interval = pd.Timedelta(minutes=LIVE_BAR_TIMEFRAME_MIN)
    publish_buffer = pd.Timedelta(seconds=15) 
    last_bar_time = None
    active_entry_bar_time = None

    print(f"Loop started for {SYMBOL}. Device: {device}")

    while True:
        try:
            market_open, next_open = is_us_equity_market_open()
            if not market_open:
                wait_secs = min(seconds_until_next_open(next_open), 30 * 60)
                if next_open is not None:
                    print(f"U.S. equity market is closed. Next open: {next_open}. Sleeping {wait_secs}s...")
                else:
                    print(f"U.S. equity market is closed. Sleeping {wait_secs}s...")
                last_bar_time = None
                time.sleep(wait_secs)
                continue

            # 1. Fetch data with explicit awareness of current time
            # Ensure your get_last_60_stock_bars uses 'end=datetime.now(timezone.utc)'
            df = get_last_60_stock_bars(SYMBOL, lookback_bars=CONTEXT_SIZE + 5)
            
            if df.empty:
                print("Warning: Received empty dataframe from API.")
                time.sleep(10)
                continue

            # Ensure index is UTC aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            bar_time = df.index[-1]
            now_utc = pd.Timestamp.now(tz="UTC")
            bar_age = now_utc - bar_time
            max_initial_bar_age = bar_interval + pd.Timedelta(minutes=20)

            if last_bar_time is None and bar_age > max_initial_bar_age:
                print(
                    f"Latest {SYMBOL} bar is stale at startup: bar_time={bar_time}, "
                    f"age={bar_age}. Waiting for a fresh bar..."
                )
                time.sleep(30)
                continue

            # 2. Check if we are looking at a stale bar (e.g., yesterday's close)
            if last_bar_time is not None and bar_time <= last_bar_time:
                # Calculate how long until the NEXT bar should exist
                next_expected_bar = last_bar_time + bar_interval
                
                if now_utc < (next_expected_bar + publish_buffer):
                    wait_secs = max(5, int(((next_expected_bar + publish_buffer) - now_utc).total_seconds()))
                    print(f"Waiting for {next_expected_bar}. Current latest: {bar_time}. Sleeping {wait_secs}s...")
                    time.sleep(wait_secs)
                else:
                    # If we are past the expected time but still getting old bars, market might be closed
                    print(f"No new data since {bar_time}. Market likely closed or illiquid. Sleeping 30s...")
                    time.sleep(30)
                continue

            # If we reached here, we have a NEW bar
            print(f"\n--- Processing New Bar: {bar_time} (UTC) ---")
            last_bar_time = bar_time

            # 3. Handle Account & Positions
            equity = get_account_equity() or starting_equity
            
            if active_entry_bar_time is not None and has_open_position(SYMBOL):
                print(f"Closing position from previous bar {active_entry_bar_time}")
                close_stock_position(SYMBOL)
                active_entry_bar_time = None

            # 4. Run Model
            side, qty, tp, sl, decision_reason = decide_trade_from_model(
                model=model,
                scaler=scaler,
                df_raw=df,
                feature_cols=feature_cols,
                context_size=CONTEXT_SIZE,
                risk_pct=DEFAULT_RISK_PCT,
                equity=equity,
                ema_span=DEFAULT_EMA_SPAN,
                momentum_weight=DEFAULT_MOMENTUM_WEIGHT,
            )

            # 5. Execute
            if side:
                order = submit_stock_bracket_order(
                    symbol=SYMBOL, side=side, qty=qty, target_price=tp, stop_price=sl
                )
                if order:
                    active_entry_bar_time = bar_time
                    post_trade_equity = get_account_equity()
                    now_utc = pd.Timestamp.now(tz="UTC").isoformat()
                    if post_trade_equity is not None:
                        print(
                            f"[EQUITY] time={now_utc} after_order side={side} "
                            f"qty={qty} equity={post_trade_equity:.2f}"
                        )
                    else:
                        print(
                            f"[EQUITY] time={now_utc} after_order side={side} "
                            f"qty={qty} equity_unavailable"
                        )
            else:
                print(f"No order submitted for this bar: {decision_reason}")
                active_entry_bar_time = None

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            time.sleep(10)

        # Standard poll sleep
        time.sleep(poll_seconds)


if __name__ == "__main__":
    live_paper_loop()
