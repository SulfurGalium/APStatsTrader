from __future__ import annotations

from typing import Optional

import pandas as pd

from config import FRED_API_KEY, MACRO_CACHE_CSV
from data.fred_cache import get_treasury_yield_history


def get_macro_enriched_data(
    csv_path: str,
    api_key: Optional[str] = None,
    cache_name: str = MACRO_CACHE_CSV,
) -> pd.DataFrame:
    """
    Load intraday OHLCV data and enrich with 10Y Treasury yield (DGS10).

    - Caches daily DGS10 series to avoid FRED rate limits.
    - Aligns daily macro to intraday timestamps via forward-fill.
    """
    api_key = api_key or FRED_API_KEY
    if not api_key:
        raise ValueError("FRED API key is missing. Set FRED_API_KEY or pass api_key.")

    df = pd.read_csv(csv_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time").sort_index()

    yield_data = get_treasury_yield_history(
        start=df.index.min(),
        end=df.index.max(),
        api_key=api_key,
        cache_name=cache_name,
    )
    yield_data = yield_data.resample("D").ffill()
    combined = df.join(yield_data, how="left").ffill().dropna()
    return combined
