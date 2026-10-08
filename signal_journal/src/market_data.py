"""OHLC e historial desde datasets del quant-fund (solo lectura)."""

from __future__ import annotations

import pandas as pd

from src.config import QUANT_FUND_ROOT


def load_price_history(symbol: str, *, interval: str = "1d", bars: int = 90) -> pd.DataFrame:
    path = QUANT_FUND_ROOT / f"data/datasets/interval={interval}/symbol={symbol}/data.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        elif "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["datetime"], utc=True))
    df = df.sort_index().tail(bars)
    return df
