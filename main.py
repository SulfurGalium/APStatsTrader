from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
import torch

from backtest.monte_carlo import tree_test
from config import (
    CONTEXT_SIZE,
    HORIZON,
    HISTORICAL_CSV,
    SIMULATION_CSV,
    MODEL_PREFIX,
    BATCH_SIZE,
    EPOCHS,
    DIFFUSION_STEPS,
    BETA_SCHEDULE,
    DEFAULT_BRANCHES,
    DEFAULT_EQUITY,
    DEFAULT_RISK_PCT,
    DEFAULT_EMA_SPAN,
    DEFAULT_MOMENTUM_WEIGHT,
)
from data.datasets import preprocess_data, TimeSeriesDataset
from data.market_data import get_macro_enriched_data
from models.diffusion import Conv1dDenoiser, TimeSeriesDiffusion
from training.io import save_model_and_scaler, load_model_and_scaler
from training.train import train_diffusion_model, tune_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion-based time series trader")

    parser.add_argument("--data", default=HISTORICAL_CSV, help="Historical OHLCV+macro CSV for training")
    parser.add_argument("--sim", default=SIMULATION_CSV, help="CSV used for simulation / backtest")
    parser.add_argument("--model-prefix", default=MODEL_PREFIX, help="Checkpoint prefix")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default=BETA_SCHEDULE)
    parser.add_argument("--branches", type=int, default=DEFAULT_BRANCHES)
    parser.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cols", default=None, help="Comma-separated list of columns to use as features")
    parser.add_argument("--ema-span", type=int, default=DEFAULT_EMA_SPAN)
    parser.add_argument("--momentum-weight", type=float, default=DEFAULT_MOMENTUM_WEIGHT)
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=5)
    parser.add_argument("--tune-batch", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--load-cosine-beta",
        action="store_true",
        help="Use cosine beta schedule when loading a saved model (default: disabled).",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    loader_workers = 0
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if args.feature_cols:
        feature_cols: List[str] = [c.strip() for c in args.feature_cols.split(",")]
    else:
        # Example default: OHLCV + UST10Y
        feature_cols = ["Open", "High", "Low", "Close", "Volume", "UST10Y"]

    context_size = args.context_size
    horizon = args.horizon

    if not args.no_train:
        # Load and preprocess the training set only when training is requested.
        train_df = get_macro_enriched_data(args.data)
        price_cols = feature_cols[:5]
        rate_cols = feature_cols[5:]
        features_scaled, scaler, _ = preprocess_data(train_df, price_cols, rate_cols)
        feature_dim = features_scaled.shape[1]
        window_dataset = TimeSeriesDataset(
            features_scaled,
            context_size=context_size,
            horizon_size=horizon,
        )
        if args.tune:
            param_grid = [
                {
                    "hidden_dim": 128,
                    "diffusion_steps": args.diffusion_steps,
                    "beta_schedule": args.beta_schedule,
                    "lr": 1e-4,
                },
                {
                    "hidden_dim": 256,
                    "diffusion_steps": args.diffusion_steps,
                    "beta_schedule": args.beta_schedule,
                    "lr": 5e-5,
                },
            ]
            best_model, best_cfg = tune_hyperparameters(
                train_features=features_scaled,
                context_size=context_size,
                horizon=horizon,
                feature_dim=feature_dim,
                device=device,
                param_grid=param_grid,
                max_epochs=args.tune_epochs,
                batch_size=args.tune_batch,
                val_frac=0.2,
                seed=args.seed,
                use_amp=args.amp,
                num_workers=loader_workers,
            )
            model = best_model
        else:
            denoiser = Conv1dDenoiser(
                feature_dim=feature_dim,
                context_size=context_size,
                horizon_size=horizon,
                hidden_dim=256,
            )
            # Use fewer diffusion steps for training speed; you can still sample with more later if desired
            train_diff_steps = min(args.diffusion_steps, 30)

            model = TimeSeriesDiffusion(
                denoiser=denoiser,
                diffusion_steps=train_diff_steps,
                beta_schedule=args.beta_schedule,
            )

            # PyTorch 2.x graph compilation for extra speed
            # DISABLE torch.compile - Triton not supported on this system
            # try:
            #     model = torch.compile(model)
            # except Exception:
            #     print("torch.compile not available or failed; continuing without compilation.")

            model = train_diffusion_model(
                dataset=window_dataset,
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-4,
                num_workers=loader_workers,
                use_amp=True if not args.amp else args.amp,  # default to AMP ON
                device=device,
            )
        save_model_and_scaler(model, scaler, args.model_prefix)
    else:
        model, scaler = load_model_and_scaler(
            args.model_prefix,
            device=device,
            use_cosine_beta=args.load_cosine_beta,
        )
        if model is None or scaler is None:
            raise RuntimeError("Failed to load model/scaler; cannot proceed with simulation.")

    if args.no_sim:
        return

    # Simulation / backtest
    sim_df = pd.read_csv(args.sim)
    sim_df["Time"] = pd.to_datetime(sim_df["Time"])
    sim_df = sim_df.set_index("Time").sort_index()

    equity_df, metrics = tree_test(
        model=model,
        scaler=scaler,
        equity=args.equity,
        branches=args.branches,
        dataset=sim_df,
        feature_cols=feature_cols,
        context_size=context_size,
        vision=args.horizon,
        seed=args.seed,
        risk_pct=args.risk_pct,
        verbose=True,
        ema_span=args.ema_span,
        momentum_weight=args.momentum_weight,
        live=args.live,
    )

    print("\nBacktest complete.")
    print(equity_df.tail())


if __name__ == "__main__":
    main()
