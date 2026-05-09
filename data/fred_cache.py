from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred

from config import FRED_API_KEY, FRED_YIELD_SERIES, MACRO_CACHE_CSV


def _normalize_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp:
    if value is None:
        ts = pd.Timestamp.utcnow()
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _read_cache(cache_name: str = MACRO_CACHE_CSV) -> pd.DataFrame:
    cache_path = Path(cache_name)
    if not cache_path.exists():
        return pd.DataFrame(columns=["UST10Y"])

    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    if "UST10Y" not in cached.columns and len(cached.columns) == 1:
        cached.columns = ["UST10Y"]
    cached.index = pd.to_datetime(cached.index)
    return cached.sort_index()


def _write_cache(df: pd.DataFrame, cache_name: str = MACRO_CACHE_CSV) -> None:
    cache_path = Path(cache_name)
    df.sort_index().to_csv(cache_path)


def get_treasury_yield_history(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    api_key: Optional[str] = None,
    cache_name: str = MACRO_CACHE_CSV,
) -> pd.DataFrame:
    start_ts = _normalize_timestamp(start)
    end_ts = _normalize_timestamp(end)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    cached = _read_cache(cache_name)
    combined = cached.copy()

    api_key = api_key or FRED_API_KEY
    should_refresh = combined.empty or combined.index.max() < end_ts or combined.index.min() > start_ts

    if api_key and should_refresh:
        try:
            if combined.empty:
                fetch_start = start_ts
            else:
                fetch_start = min(start_ts, combined.index.max() - pd.Timedelta(days=7))
            fred = Fred(api_key=api_key)
            fetched = fred.get_series(
                FRED_YIELD_SERIES,
                observation_start=fetch_start.date(),
                observation_end=end_ts.date(),
            )
            fetched_df = fetched.to_frame(name="UST10Y").dropna()
            if not fetched_df.empty:
                merged = pd.concat([combined, fetched_df]).sort_index()
                combined = merged.loc[~merged.index.duplicated(keep="last")]
                _write_cache(combined, cache_name)
        except Exception as exc:
            if combined.empty:
                raise RuntimeError(f"Unable to fetch FRED data and no cache is available: {exc}") from exc

    if combined.empty:
        raise RuntimeError("No macro yield data available from FRED or local cache.")

    window = combined.loc[(combined.index >= start_ts) & (combined.index <= end_ts)]
    if window.empty:
        return combined
    return window


def get_latest_treasury_yield(
    at_time: pd.Timestamp | str | None = None,
    api_key: Optional[str] = None,
    cache_name: str = MACRO_CACHE_CSV,
) -> float:
    target_ts = _normalize_timestamp(at_time)
    history = get_treasury_yield_history(
        start=target_ts - pd.Timedelta(days=30),
        end=target_ts,
        api_key=api_key,
        cache_name=cache_name,
    )
    eligible = history.loc[history.index <= target_ts, "UST10Y"].dropna()
    if eligible.empty:
        eligible = history["UST10Y"].dropna()
    if eligible.empty:
        raise RuntimeError("No usable treasury yield values were found in cache or FRED.")
    return float(eligible.iloc[-1])
