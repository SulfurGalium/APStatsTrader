from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_NUM_SAMPLES
from models.diffusion import TimeSeriesDiffusion
from trading.signal_engine import build_processed_features, derive_trade_plan


def batched_monte_carlo_backtest(
    model: TimeSeriesDiffusion,
    scaler,
    dataset_features: np.ndarray,
    dataset_raw: pd.DataFrame,
    branches: int,
    context_size: int,
    horizon_steps: int = 10,
    initial_equity: float = 10_000.0,
    risk_pct: float = 0.02,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: Optional[int] = None,
    ema_span: int = 20,
    momentum_weight: float = 0.5,
    verbose: bool = False,
    live: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo backtest using one-bar trade resolution.
    """
    del verbose
    device = next(model.parameters()).device

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    max_start = len(dataset_features) - context_size - horizon_steps - 1
    if max_start < 0:
        raise ValueError("Not enough data to run backtest.")

    if live:
        start_indices = np.array([0])
        branches = 1
    else:
        start_indices = np.random.randint(0, max_start + 1, size=branches)

    feature_dim = dataset_features.shape[1]
    equity_curves = np.ones((horizon_steps + 1, branches), dtype=np.float32) * initial_equity
    forecast_horizon = max(1, int(getattr(model.denoiser, "horizon_size", 1)))
    close_scale = float(scaler.scale_[3])
    close_mean = float(scaler.mean_[3])

    trade_counts = np.zeros(branches, dtype=int)
    win_counts = np.zeros(branches, dtype=int)
    confidence_sums = np.zeros(branches, dtype=float)

    close_ema = dataset_raw["Close"].ewm(span=ema_span, adjust=False).mean().values

    for step in range(horizon_steps):
        contexts = []
        for idx in start_indices:
            ctx_start = idx + step
            ctx_end = ctx_start + context_size
            contexts.append(dataset_features[ctx_start:ctx_end])

        batch_context = torch.tensor(np.array(contexts), dtype=torch.float32, device=device)
        predictions = model.sample(
            batch_context,
            horizon=forecast_horizon,
            feature_dim=feature_dim,
            num_samples=num_samples,
        )
        close_return_paths = predictions[:, :, 0, 3].cpu().numpy()
        unscaled_close_paths = close_return_paths * close_scale + close_mean
        mu = np.median(unscaled_close_paths, axis=1)
        sigma = np.std(unscaled_close_paths, axis=1)

        for b, start_idx in enumerate(start_indices):
            current_idx = start_idx + context_size + step - 1
            next_idx = start_idx + context_size + step

            current_price = float(dataset_raw.iloc[current_idx]["Close"])
            next_bar = dataset_raw.iloc[next_idx]
            next_close = float(next_bar["Close"])
            next_low = float(next_bar["Low"])
            next_high = float(next_bar["High"])

            ema_val = float(close_ema[current_idx])
            recent_closes = dataset_raw["Close"].iloc[start_idx + step : start_idx + step + context_size]

            trade_plan = derive_trade_plan(
                mu=float(mu[b]),
                sigma=float(sigma[b]),
                current_price=current_price,
                recent_closes=recent_closes,
                ema_val=ema_val,
                equity=float(equity_curves[step, b]),
                risk_pct=risk_pct,
                momentum_weight=momentum_weight,
            )

            if trade_plan is None:
                equity_curves[step + 1, b] = equity_curves[step, b]
                continue

            trade_counts[b] += 1
            confidence_sums[b] += trade_plan.confidence_discount

            pnl = 0.0
            if trade_plan.side == "buy":
                hit_stop = next_low <= trade_plan.stop_price
                hit_target = next_high >= trade_plan.target_price
                if hit_stop and hit_target:
                    if next_close >= trade_plan.current_price:
                        pnl = trade_plan.qty * (trade_plan.target_price - trade_plan.current_price)
                        win_counts[b] += 1
                    else:
                        pnl = -trade_plan.qty * trade_plan.stop_distance
                elif hit_target:
                    pnl = trade_plan.qty * (trade_plan.target_price - trade_plan.current_price)
                    win_counts[b] += 1
                elif hit_stop:
                    pnl = -trade_plan.qty * trade_plan.stop_distance
                else:
                    pnl = trade_plan.qty * (next_close - trade_plan.current_price)
                    if pnl > 0:
                        win_counts[b] += 1
            else:
                hit_stop = next_high >= trade_plan.stop_price
                hit_target = next_low <= trade_plan.target_price
                if hit_stop and hit_target:
                    if next_close <= trade_plan.current_price:
                        pnl = trade_plan.qty * (trade_plan.current_price - trade_plan.target_price)
                        win_counts[b] += 1
                    else:
                        pnl = -trade_plan.qty * trade_plan.stop_distance
                elif hit_target:
                    pnl = trade_plan.qty * (trade_plan.current_price - trade_plan.target_price)
                    win_counts[b] += 1
                elif hit_stop:
                    pnl = -trade_plan.qty * trade_plan.stop_distance
                else:
                    pnl = trade_plan.qty * (trade_plan.current_price - next_close)
                    if pnl > 0:
                        win_counts[b] += 1

            equity_curves[step + 1, b] = equity_curves[step, b] + pnl

    hit_ratios = np.divide(
        win_counts,
        trade_counts,
        out=np.zeros_like(win_counts, dtype=float),
        where=trade_counts != 0,
    )
    confidences = np.divide(
        confidence_sums,
        trade_counts,
        out=np.zeros_like(confidence_sums),
        where=trade_counts != 0,
    )

    equity_df = pd.DataFrame(
        equity_curves,
        index=range(horizon_steps + 1),
        columns=[f"Branch_{i+1}" for i in range(branches)],
    )
    return equity_df, trade_counts, hit_ratios, confidences


def _summary_stats(values: pd.Series | np.ndarray) -> Dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
    }


def compute_backtest_metrics(
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, pd.Series | float]:
    """
    Compute branch-level metrics plus aggregate mean/median/std summaries.
    """
    returns = equity_df.pct_change().dropna(how="all")
    final_equity = equity_df.iloc[-1]
    total_return = final_equity / equity_df.iloc[0] - 1

    if returns.empty:
        metrics: Dict[str, pd.Series | float] = {
            "final_equity": final_equity,
            "total_return": total_return,
            "sharpe": pd.Series(0.0, index=equity_df.columns),
            "max_drawdown": pd.Series(0.0, index=equity_df.columns),
            "win_rate": pd.Series(0.0, index=equity_df.columns),
        }
    else:
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=0).replace(0, np.nan)
        sharpe = ((mean_ret - risk_free_rate) / std_ret).fillna(0.0)

        drawdown = equity_df.div(equity_df.cummax()) - 1
        max_drawdown = drawdown.min()

        win_rate = ((returns > 0).sum() / returns.count()).fillna(0.0)

        metrics = {
            "final_equity": final_equity,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    for metric_name in ["final_equity", "total_return", "sharpe", "max_drawdown", "win_rate"]:
        stats = _summary_stats(metrics[metric_name])
        metrics[f"{metric_name}_mean"] = stats["mean"]
        metrics[f"{metric_name}_median"] = stats["median"]
        metrics[f"{metric_name}_std"] = stats["std"]

    return metrics


def tree_test(
    model: TimeSeriesDiffusion,
    scaler,
    equity: float,
    branches: int,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    context_size: int,
    vision: Optional[int] = None,
    seed: Optional[int] = None,
    risk_pct: float = 0.02,
    verbose: bool = False,
    ema_span: int = 20,
    momentum_weight: float = 0.8,
    live: bool = False,
):
    """
    Wrapper that builds features and runs Monte Carlo backtest.
    """
    features, aligned_raw = build_processed_features(dataset, feature_cols)
    scaled_features = scaler.transform(features)

    horizon = int(round(vision)) if vision is not None else 1
    horizon = max(1, horizon)

    max_possible = max(1, len(scaled_features) - context_size - 1)
    if horizon > max_possible:
        horizon = max_possible
        if verbose:
            print(f"Horizon too large; reduced to {horizon}.")

    if live:
        branches = 1
        horizon = max_possible
        print(f"Live mode: simulating single branch through {horizon} steps.")

    equity_df, trade_counts, hit_ratios, confidences = batched_monte_carlo_backtest(
        model=model,
        scaler=scaler,
        dataset_features=scaled_features,
        dataset_raw=aligned_raw,
        branches=branches,
        context_size=context_size,
        horizon_steps=horizon,
        initial_equity=equity,
        num_samples=DEFAULT_NUM_SAMPLES,
        seed=seed,
        risk_pct=risk_pct,
        ema_span=ema_span,
        momentum_weight=momentum_weight,
        verbose=verbose,
        live=live,
    )

    metrics = compute_backtest_metrics(equity_df)
    metrics["trade_count_mean"] = float(np.mean(trade_counts))
    metrics["trade_count_median"] = float(np.median(trade_counts))
    metrics["trade_count_std"] = float(np.std(trade_counts, ddof=0))
    metrics["confidence_mean"] = float(np.mean(confidences))
    metrics["confidence_median"] = float(np.median(confidences))
    metrics["confidence_std"] = float(np.std(confidences, ddof=0))

    if not live:
        print("\n=== Per-Branch Stats ===")
        for i in range(branches):
            print(
                f"Branch {i+1}: trades={trade_counts[i]}, "
                f"hit_ratio={hit_ratios[i]:.3f}, confidence={confidences[i]:.3f}"
            )
        print("\n=== Aggregate Metrics ===")
    else:
        print("\n=== Live-Faithful Simulation Metrics ===")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return equity_df, metrics
