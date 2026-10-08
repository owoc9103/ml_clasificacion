"""
Cross-asset features for improved prediction.

These features capture market regime and relationships between assets:
- VIX term structure (stress indicator)
- Bond-stock correlation (regime shifts)
- Credit spreads (risk appetite)
- Relative strength (outperformance vs market)
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd

Interval = Literal["1d", "5d", "60m", "120m"]


def _load_symbol_close(parquet_dir: str, interval: Interval, symbol: str) -> pd.Series:
    """Load close prices for a symbol."""
    path = os.path.join(parquet_dir, f"interval={interval}", f"symbol={symbol}", "data.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    if "close" not in df.columns:
        return pd.Series(dtype=float, index=df.index)

    return df["close"].astype(float)


def compute_cross_asset_features(
    symbol: str,
    interval: Interval,
    parquet_dir: str = "data/parquet",
) -> pd.DataFrame:
    """
    Compute cross-asset features for a symbol.

    Features:
    - vix_level: VIX level (fear gauge)
    - vix_trend: VIX 5-day vs 20-day MA (rising vol = stress)
    - bond_stock_corr: 20-day rolling correlation TLT vs SPY
    - credit_spread: HYG/TLT ratio (high = risk-on, low = risk-off)
    - relative_strength: symbol return vs SPY return over 20 days
    - spy_trend: SPY above/below 200-day MA (bull/bear market)
    """
    # Load required asset prices
    sym_close = _load_symbol_close(parquet_dir, interval, symbol)
    if sym_close.empty:
        return pd.DataFrame()

    spy_close = _load_symbol_close(parquet_dir, interval, "SPY")
    vix_close = _load_symbol_close(parquet_dir, interval, "^VIX")
    tlt_close = _load_symbol_close(parquet_dir, interval, "TLT")
    hyg_close = _load_symbol_close(parquet_dir, interval, "HYG")

    # Align all to symbol index
    spy_close = spy_close.reindex(sym_close.index).ffill()
    vix_close = vix_close.reindex(sym_close.index).ffill()
    tlt_close = tlt_close.reindex(sym_close.index).ffill()
    hyg_close = hyg_close.reindex(sym_close.index).ffill()

    # VIX features
    vix_level = vix_close
    vix_ma5 = vix_close.rolling(5).mean()
    vix_ma20 = vix_close.rolling(20).mean()
    vix_trend = (vix_ma5 / vix_ma20) - 1.0  # Rising VIX = positive, falling = negative

    # Bond-stock correlation (regime indicator)
    spy_ret = spy_close.pct_change()
    tlt_ret = tlt_close.pct_change()
    bond_stock_corr = spy_ret.rolling(20).corr(tlt_ret)

    # Credit spread proxy (HYG vs TLT)
    # High HYG/TLT = risk-on (credit outperforms treasuries)
    # Low HYG/TLT = risk-off (flight to quality)
    credit_spread = (hyg_close / tlt_close).pct_change(20)

    # Relative strength vs SPY
    sym_ret_20 = sym_close.pct_change(20)
    spy_ret_20 = spy_close.pct_change(20)
    relative_strength = sym_ret_20 - spy_ret_20

    # SPY trend (bull/bear market indicator)
    spy_sma200 = spy_close.rolling(200).mean()
    spy_trend = (spy_close > spy_sma200).astype(float)

    # Market breadth proxy: SPY distance from highs
    spy_high_52w = spy_close.rolling(min(252, len(spy_close))).max()
    market_breadth = (spy_close - spy_high_52w) / spy_high_52w

    feat = pd.DataFrame({
        "vix_level": vix_level,
        "vix_trend": vix_trend,
        "bond_stock_corr": bond_stock_corr,
        "credit_spread": credit_spread,
        "relative_strength": relative_strength,
        "spy_trend": spy_trend,
        "market_breadth": market_breadth,
    }, index=sym_close.index)

    return feat.replace([np.inf, -np.inf], np.nan)


__all__ = ["compute_cross_asset_features"]
