from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_NUM_SAMPLES,
    LIVE_STOP_MULTIPLIER,
    MAX_POSITION_NOTIONAL,
    MIN_EDGE_TO_STOP_RATIO,
    MIN_LOCAL_VOL,
    REQUIRE_TREND_CONFIRMATION,
    SIGNAL_TO_NOISE_THRESHOLD,
)


@dataclass
class TradePlan:
    mu: float
    sigma: float
    predicted_close: float
    confidence_discount: float
    signal_to_noise: float
    edge_to_stop_ratio: float
    stop_distance: float
    qty: int
    side: str
    target_price: float
    stop_price: float
    current_price: float


def build_processed_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    price_cols = feature_cols[:5]
    rate_cols = feature_cols[5:]

    eps = 1e-8
    processed = pd.DataFrame(index=df.index)
    for col in price_cols:
        processed[col] = np.log((df[col] + eps) / (df[col].shift(1) + eps))
    for col in rate_cols:
        processed[col] = df[col] - df[col].shift(1)

    processed = processed.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_raw = df.loc[processed.index]
    features = processed[feature_cols].values
    return features, aligned_raw


@torch.no_grad()
def sample_return_distribution(
    model,
    context: np.ndarray,
    scaler,
    feature_dim: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    forecast_horizon = max(1, int(getattr(model.denoiser, "horizon_size", 1)))
    context_tensor = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
    preds = model.sample(
        context_tensor,
        horizon=forecast_horizon,
        feature_dim=feature_dim,
        num_samples=num_samples,
    )
    close_return_paths = preds[0, :, 0, 3].cpu().numpy()
    close_scale = float(scaler.scale_[3])
    close_mean = float(scaler.mean_[3])
    unscaled_close_paths = close_return_paths * close_scale + close_mean
    return float(np.median(unscaled_close_paths)), float(np.std(unscaled_close_paths))


def derive_trade_plan(
    *,
    mu: float,
    sigma: float,
    current_price: float,
    recent_closes: pd.Series,
    ema_val: float,
    equity: float,
    risk_pct: float,
    momentum_weight: float,
    stop_multiplier: float = LIVE_STOP_MULTIPLIER,
) -> Optional[TradePlan]:
    # Express momentum in percentage terms so CLI changes to momentum_weight have a real effect.
    momentum_pct = 0.0 if ema_val == 0 else ((current_price - ema_val) / (ema_val + 1e-9)) * 100.0
    momentum_bias = float(np.tanh(momentum_pct * momentum_weight))
    mu = float(np.clip(mu, -0.35, 0.35))
    sigma = max(float(sigma), 1e-6)

    predicted_close = current_price * np.exp(mu)
    predicted_close = current_price + (predicted_close - current_price) * (1 + momentum_bias)

    local_vol = float(recent_closes.std(ddof=0))
    if np.isnan(local_vol) or local_vol <= 1e-8:
        local_vol = MIN_LOCAL_VOL
    else:
        local_vol = max(local_vol, MIN_LOCAL_VOL)

    stop_distance = float(max(local_vol * stop_multiplier, 0.10))
    predicted_move = abs(predicted_close - current_price)
    edge_to_stop_ratio = predicted_move / max(stop_distance, 1e-6)
    signal_to_noise = abs(mu) / sigma

    if signal_to_noise < SIGNAL_TO_NOISE_THRESHOLD:
        return None
    if edge_to_stop_ratio < MIN_EDGE_TO_STOP_RATIO:
        return None

    side = "buy" if predicted_close > current_price else "sell"
    trend_side = "buy" if current_price >= ema_val else "sell"
    if REQUIRE_TREND_CONFIRMATION and side != trend_side:
        return None

    confidence_discount = float(
        np.clip(signal_to_noise / (SIGNAL_TO_NOISE_THRESHOLD * 2.0), 0.05, 1.0)
    )
    confidence_discount *= float(np.clip(edge_to_stop_ratio / (MIN_EDGE_TO_STOP_RATIO * 2.0), 0.5, 1.0))
    confidence_discount *= 1 + abs(momentum_bias) * 0.25

    adjusted_risk = equity * risk_pct * confidence_discount
    adjusted_risk = min(adjusted_risk, MAX_POSITION_NOTIONAL)

    qty = int(adjusted_risk / stop_distance)
    if qty <= 0 and adjusted_risk >= stop_distance * 0.1:
        qty = 1

    if qty <= 0:
        return None

    qty = min(qty, 2_000)
    if qty <= 0:
        return None

    if side == "buy":
        target_price = current_price + stop_distance * 2.0
        stop_price = current_price - stop_distance
    else:
        target_price = current_price - stop_distance * 2.0
        stop_price = current_price + stop_distance

    return TradePlan(
        mu=mu,
        sigma=sigma,
        predicted_close=float(predicted_close),
        confidence_discount=confidence_discount,
        signal_to_noise=signal_to_noise,
        edge_to_stop_ratio=edge_to_stop_ratio,
        stop_distance=stop_distance,
        qty=qty,
        side=side,
        target_price=float(target_price),
        stop_price=float(stop_price),
        current_price=current_price,
    )
