"""
Data ingestion via yfinance for ETFs and indices.

- Supports EOD (1d) and intraday (60m, 120m)
- Normalizes timestamps to UTC, no look-ahead
- Handles actions via auto-adjusted OHLC
- Writes Parquet partitioned by interval/symbol with zstd compression

Usage:
    from quantfund.data.ingest import ingest_symbols
    ingest_symbols(["SPY","QQQ"],["1d","60m"], start="2012-01-01", end="2025-09-27")
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from quantfund.utils.log import get_logger

logger = get_logger(__name__)

Interval = Literal["1d", "5d", "60m", "120m"]


def _normalize_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    df.index.name = "datetime"
    return df


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    # Build mapping lower->original then invert to map original->lower
    inv = {v: k for k, v in cols.items()}
    out = df.rename(columns=inv)
    # Standardize expected columns
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adjclose",
        "volume": "volume",
        "dividends": "dividends",
        "stock_splits": "splits",
    }
    out = out.rename(columns=rename_map)
    return out


def _yf_history(symbol: str, interval: Interval, start: str, end: str) -> pd.DataFrame:
    # yfinance intervals: supports "1d", "5d", "60m"; "120m" is not standard, resample from 60m
    if interval in ("60m", "120m"):
        yf_interval = "60m"
    elif interval == "5d":
        yf_interval = "1wk"  # yfinance uses "1wk" for weekly data
    else:
        yf_interval = "1d"

    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start, end=end, interval=yf_interval, auto_adjust=True, actions=True)
    df = _normalize_utc_index(df)
    df = _rename_ohlcv(df)
    if interval == "120m" and not df.empty:
        # Resample 120m on UTC index using OHLCV aggregation
        o = df["open"].resample("120T").first()
        h = df["high"].resample("120T").max()
        l = df["low"].resample("120T").min()
        c = df["close"].resample("120T").last()
        v = df["volume"].resample("120T").sum().astype(float)
        d = df.get("dividends")
        s = df.get("splits")
        data = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        if d is not None:
            data["dividends"] = d.resample("120T").sum()
        if s is not None:
            data["splits"] = s.resample("120T").sum()
        df = pd.DataFrame(data).dropna(subset=["close"])  # ensure valid bars
    return df


def fetch_symbol(symbol: str, interval: Interval, start: str, end: str) -> pd.DataFrame:
    df = _yf_history(symbol, interval, start, end)
    if df.empty:
        return df
    df = df.assign(symbol=symbol, interval=interval)
    # Ensure types
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df


def _partition_path(base_dir: str, interval: Interval, symbol: str) -> str:
    return os.path.join(base_dir, f"interval={interval}", f"symbol={symbol}")


def save_parquet(df: pd.DataFrame, base_dir: str, interval: Interval, symbol: str) -> str:
    part_dir = _partition_path(base_dir, interval, symbol)
    os.makedirs(part_dir, exist_ok=True)
    path = os.path.join(part_dir, "data.parquet")
    df.to_parquet(path, engine="pyarrow", compression="zstd")
    return path


def ingest_symbols(
    symbols: Iterable[str],
    intervals: Iterable[Interval],
    start: str,
    end: str,
    out_dir: str = "data/parquet",
) -> list[str]:
    """Fetch and persist OHLCV for a list of symbols and intervals.

    Returns list of written file paths.
    """
    written: list[str] = []
    for symbol in symbols:
        for interval in intervals:
            try:
                logger.info(f"Fetching {symbol} {interval} from {start} to {end}")
                df = fetch_symbol(symbol, interval, start, end)
                if df.empty:
                    logger.warning(f"No data returned for {symbol} {interval}")
                    # Write empty schema-aligned file
                    df = pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume", "dividends", "splits", "symbol", "interval"],
                        index=pd.DatetimeIndex([], name="datetime"),
                    )
                else:
                    logger.info(f"Fetched {len(df)} rows for {symbol} {interval}")
                written.append(save_parquet(df, out_dir, interval, symbol))
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} {interval}: {type(e).__name__}: {e}")
                empty = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume", "dividends", "splits", "symbol", "interval"],
                    index=pd.DatetimeIndex([], name="datetime"),
                )
                written.append(save_parquet(empty, out_dir, interval, symbol))
    return written


__all__ = [
    "fetch_symbol",
    "ingest_symbols",
    "save_parquet",
]