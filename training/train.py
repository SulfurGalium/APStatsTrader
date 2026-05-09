from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.datasets import TimeSeriesDataset
from models.diffusion import Conv1dDenoiser, TimeSeriesDiffusion


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def build_window_tensor(features, context, horizon):
    X = []
    Y = []
    for i in range(len(features) - context - horizon):
        X.append(features[i:i+context])
        Y.append(features[i+context:i+context+horizon])
    return torch.tensor(X), torch.tensor(Y)

def train_diffusion_model(
    dataset: torch.utils.data.Dataset,
    model: TimeSeriesDiffusion,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
    use_amp: bool = True,
    device: torch.device | None = None,
) -> TimeSeriesDiffusion:
    """
    Train diffusion model on windowed time-series dataset.

    Optimizations:
      - pin_memory + persistent_workers + prefetch_factor
      - AMP enabled by default
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # prefetch_factor must be None when num_workers = 0
    pf = 4 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=pf,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=10, min_delta=1e-6)

    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_device = device.type if device.type in ["cuda", "cpu"] else "cpu"

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, enabled=use_amp):
                loss = model(y, x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        if early_stop(avg_loss):
            print("Early stopping triggered.")
            break

    return model


def evaluate_diffusion_model(
    model: TimeSeriesDiffusion,
    data_loader: DataLoader,
    device: torch.device | None = None,
) -> float:
    """
    Compute average MSE loss on validation dataset.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss = model(y, x)
            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

    return total_loss / max(1, total_samples)


def tune_hyperparameters(
    train_features: np.ndarray,
    context_size: int,
    horizon: int,
    feature_dim: int,
    device: torch.device,
    param_grid: List[Dict],
    max_epochs: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.2,
    seed: Optional[int] = None,
    use_amp: bool = False,
    num_workers: int = 0,
) -> Tuple[TimeSeriesDiffusion, Dict]:
    """
    Simple grid search over hyperparameters using validation loss.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    total = len(train_features)
    val_start = int(total * (1 - val_frac))
    train_arr = train_features[:val_start]
    val_arr = train_features[val_start:]

    best_loss = float("inf")
    best_config: Dict | None = None
    best_model: TimeSeriesDiffusion | None = None

    for cfg in param_grid:
        denoiser = Conv1dDenoiser(
            feature_dim=feature_dim,
            context_size=context_size,
            horizon_size=horizon,
            hidden_dim=cfg.get("hidden_dim", 128),
        ).to(device)

        model = TimeSeriesDiffusion(
            denoiser,
            diffusion_steps=cfg.get("diffusion_steps", 50),
            beta_schedule=cfg.get("beta_schedule", "linear"),
        ).to(device)

        train_ds = TimeSeriesDataset(train_arr, context_size, horizon)
        val_ds = TimeSeriesDataset(val_arr, context_size, horizon)

        trained = train_diffusion_model(
            train_ds,
            model,
            epochs=max_epochs,
            batch_size=batch_size,
            lr=cfg.get("lr", 1e-4),
            num_workers=num_workers,
            use_amp=use_amp,
            device=device,
        )

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loss = evaluate_diffusion_model(trained, val_loader, device=device)

        print(f"Tuning {cfg} -> val loss {loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            best_config = cfg
            best_model = trained

    if best_model is None or best_config is None:
        raise RuntimeError("Hyperparameter tuning failed to produce a model.")

    print(f"Best hyperparams: {best_config} (val_loss={best_loss:.6f})")
    return best_model, best_config