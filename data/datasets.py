from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


def build_window_tensors(
    features: np.ndarray,
    context_size: int,
    horizon_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute all (context, target) windows into contiguous tensors.

    This is still safe with your data size and 4 GB RAM.
    """
    X = []
    Y = []
    n = len(features) - context_size - horizon_size
    for i in range(n):
        X.append(features[i : i + context_size])
        Y.append(features[i + context_size : i + context_size + horizon_size])
    X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    return X, Y

class TimeSeriesDataset(Dataset):
    """
    Lazy windowed time-series dataset.

    Each item:
        x: context window  (context_size, feature_dim)
        y: target window   (horizon_size, feature_dim)
    """

    def __init__(self, features: np.ndarray, context_size: int, horizon_size: int) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.context_size = context_size
        self.horizon_size = horizon_size

    def __len__(self) -> int:
        return len(self.features) - self.context_size - self.horizon_size

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.context_size]
        y = self.features[idx + self.context_size : idx + self.context_size + self.horizon_size]
        return x, y


def preprocess_data(
    df: pd.DataFrame,
    price_cols: list[str],
    rate_cols: list[str],
    train_split: float = 0.8,
) -> Tuple[np.ndarray, StandardScaler, int]:
    """
    Transform raw OHLCV+macro into stationary features.

    - Prices: log returns
    - Rates/yields: first differences
    - Scaling: StandardScaler fit on train split only
    """
    processed_df = pd.DataFrame(index=df.index)
    eps = 1e-8

    for col in price_cols:
        processed_df[col] = np.log((df[col] + eps) / (df[col].shift(1) + eps))

    for col in rate_cols:
        processed_df[col] = df[col] - df[col].shift(1)

    processed_df = processed_df.replace([np.inf, -np.inf], np.nan).dropna()
    features = processed_df[price_cols + rate_cols].values

    split_idx = int(len(features) * train_split)
    train_features = features[:split_idx]

    scaler = StandardScaler()
    scaler.fit(train_features)
    features_scaled = scaler.transform(features)

    return features_scaled, scaler, split_idx