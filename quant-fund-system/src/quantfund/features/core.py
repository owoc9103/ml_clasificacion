"""
Feature engineering for ETF datasets.

- Loads OHLCV Parquet per symbol/interval (partitioned directories)
- Computes momentum: r_5/10/20, MACD, ADX, Sharpe20
- Computes mean-reversion: RSI2/14, EMA20 z-score, Bollinger %B
- Adds overnight gap z-score, microstructure: intraday range, volume spike z, autocorr1/5
- Regime features (can be used downstream): SPY EMA50>200 & VIX<25 and placeholder HMM flag

Strictly uses information available at bar close; no look-ahead.
Writes per-symbol features to data/datasets/interval=XX/symbol=SYM/data.parquet (zstd).

Example:
    from quantfund.features.core import build_features_dataset
    build_features_dataset(["SPY","QQQ"],["60m"], parquet_dir="data/parquet", out_dir="data/datasets")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, cast

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from quantfund.features.cross_asset import compute_cross_asset_features
from quantfund.features.fracdiff import frac_diff_ffd

Interval = Literal["1d", "5d", "60m", "120m"]


@dataclass(frozen=True)
class FeatureSpec:
    interval: Interval
    parquet_dir: str = "data/parquet"
    out_dir: str = "data/datasets"


def _parquet_path(parquet_dir: str, interval: Interval, symbol: str) -> str:
    return os.path.join(parquet_dir, f"interval={interval}", f"symbol={symbol}", "data.parquet")


def _out_path(out_dir: str, interval: Interval, symbol: str) -> str:
    return os.path.join(out_dir, f"interval={interval}", f"symbol={symbol}", "data.parquet")


def _load_symbol_frame(parquet_dir: str, interval: Interval, symbol: str) -> pd.DataFrame:
    path = _parquet_path(parquet_dir, interval, symbol)
    if not os.path.exists(path):
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _ann_factor(interval: Interval) -> float:
    if interval == "1d":
        return 252.0
    if interval == "5d":
        return 52.0  # 52 weeks per year
    if interval == "60m":
        return 252.0 * 6.5
    if interval == "120m":
        return 252.0 * 3.25
    return 252.0


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0.0, np.nan)
    return (s - m) / sd


def _autocorr(series: pd.Series, lag: int) -> pd.Series:
    res = series.rolling(lag + 1).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)
    return cast(pd.Series, res)


def _compute_features(df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    if df.empty:
        return df
    close = cast(pd.Series, df["close"]).astype(float)
    open_ = cast(pd.Series, df["open"]).astype(float)
    high = cast(pd.Series, df["high"]).astype(float)
    low = cast(pd.Series, df["low"]).astype(float)
    volume = (cast(pd.Series, df["volume"]) if "volume" in df else pd.Series(0.0, index=df.index)).astype(float)

    ret1 = close.pct_change()
    # Momentum returns
    r_5 = close.pct_change(5)
    r_10 = close.pct_change(10)
    r_20 = close.pct_change(20)

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)

    # ADX
    adx = ADXIndicator(high=high, low=low, close=close, window=14)

    # Sharpe20 (rolling mean/std of 1-bar returns)
    ann = np.sqrt(_ann_factor(interval))
    sharpe20_base = cast(pd.Series, ret1.rolling(20).mean() / ret1.rolling(20).std())
    sharpe20 = sharpe20_base.replace([np.inf, -np.inf], np.nan) * ann

    # Mean reversion features
    rsi2 = RSIIndicator(close=close, window=2).rsi()
    rsi14 = RSIIndicator(close=close, window=14).rsi()
    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema20_z = _zscore(close - ema20, window=20)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    percent_b = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    # Overnight gap z-score
    gap = (open_ / close.shift(1)) - 1.0
    gap_z = _zscore(gap, window=20)

    # Microstructure
    intraday_range = (high - low) / close
    vol_z = _zscore(volume, window=20)
    # Autocorrelation features can be numerically unstable on tiny windows and often yield NaN.
    # Fill NaNs with 0 so we don't drop all rows downstream.
    ac1 = _autocorr(ret1.fillna(0.0), lag=1).fillna(0.0)
    ac5 = _autocorr(ret1.fillna(0.0), lag=5).fillna(0.0)

    # Volatility
    atr_14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Additional momentum features
    r_3 = close.pct_change(3)
    r_30 = close.pct_change(30)
    r_60 = close.pct_change(60)

    # Momentum strength - compare short vs long term
    mom_strength = r_10 / r_30.replace(0, np.nan)  # Short-term relative to medium-term

    # Volume features
    vol_sma_20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma_20.replace(0, np.nan)  # Current vol vs average

    # Price position relative to moving averages
    sma_50 = SMAIndicator(close=close, window=50).sma_indicator()
    sma_200 = SMAIndicator(close=close, window=200).sma_indicator()
    price_to_sma50 = (close / sma_50) - 1.0  # % above/below 50-day MA
    price_to_sma200 = (close / sma_200) - 1.0  # % above/below 200-day MA

    # Volatility ratios
    vol_20 = ret1.rolling(20).std()
    vol_60 = ret1.rolling(60).std()
    vol_ratio_term = vol_20 / vol_60.replace(0, np.nan)  # Short-term vol vs long-term

    # High-low range relative to close
    hl_pct = (high - low) / close
    hl_pct_ma = hl_pct.rolling(20).mean()

    # Distance from highs/lows
    high_52w = high.rolling(min(252, len(high))).max()
    low_52w = low.rolling(min(252, len(low))).min()
    dist_from_high = (close - high_52w) / high_52w
    dist_from_low = (close - low_52w) / low_52w

    # Trend strength (days above/below MA)
    above_ema20 = (close > ema20).astype(float).rolling(20).sum() / 20.0

    # Fractional differentiation features (Phase 3)
    # Addresses non-stationarity while preserving memory
    fracdiff_d30 = frac_diff_ffd(close, d=0.3, threshold=1e-5)
    fracdiff_d40 = frac_diff_ffd(close, d=0.4, threshold=1e-5)
    fracdiff_d50 = frac_diff_ffd(close, d=0.5, threshold=1e-5)

    feat = pd.DataFrame(
        {
            # Momentum
            "r_3": r_3,
            "r_5": r_5,
            "r_10": r_10,
            "r_20": r_20,
            "r_30": r_30,
            "r_60": r_60,
            "macd": macd.macd(),
            "macd_signal": macd.macd_signal(),
            "macd_hist": macd.macd_diff(),
            "adx_14": adx.adx(),
            "sharpe_20": sharpe20,
            "mom_strength": mom_strength,
            # Mean reversion
            "rsi_2": rsi2,
            "rsi_14": rsi14,
            "ema20_z": ema20_z,
            "%b": percent_b,
            "price_to_sma50": price_to_sma50,
            "price_to_sma200": price_to_sma200,
            "above_ema20": above_ema20,
            # Overnight / micro
            "gap_z": gap_z,
            "range_intraday": intraday_range,
            "hl_pct_ma": hl_pct_ma,
            # Volume
            "vol_z": vol_z,
            "vol_ratio": vol_ratio,
            "autocorr_1": ac1,
            "autocorr_5": ac5,
            # Volatility
            "atr_14": atr_14,
            "vol_ratio_term": vol_ratio_term,
            # Position
            "dist_from_high": dist_from_high,
            "dist_from_low": dist_from_low,
            # Fractional differentiation (Phase 3)
            "fracdiff_d30": fracdiff_d30,
            "fracdiff_d40": fracdiff_d40,
            "fracdiff_d50": fracdiff_d50,
        },
        index=df.index,
    )

    # Include base cols for downstream
    feat["open"] = open_
    feat["high"] = high
    feat["low"] = low
    feat["close"] = close
    feat["volume"] = volume

    # Next-bar references for labeling; ensure no look-ahead in features
    feat["next_open"] = open_.shift(-1)
    feat["next_close"] = close.shift(-1)

    # Replace inf with NaN
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Selective imputation instead of aggressive dropna()
    # Technical indicators can be forward-filled (they persist until new data)
    feat = feat.ffill(limit=5)  # Forward fill up to 5 bars

    # Fill remaining NaNs with 0 (neutral for z-scores, safe for most features)
    # Exclude label columns from filling
    label_cols = ['next_open', 'next_close']
    feature_cols = [c for c in feat.columns if c not in label_cols]
    feat[feature_cols] = feat[feature_cols].fillna(0)

    # Only drop rows where LABELS are missing (last row typically)
    feat = feat.dropna(subset=label_cols)

    return feat


def build_features_for_symbol(symbol: str, interval: Interval, parquet_dir: str) -> pd.DataFrame:
    raw = _load_symbol_frame(parquet_dir, interval, symbol)
    if raw.empty:
        return raw

    # Compute base technical features
    feat = _compute_features(raw, interval)

    # Add cross-asset features
    cross_feat = compute_cross_asset_features(symbol, interval, parquet_dir)
    if not cross_feat.empty:
        # Merge on index
        feat = feat.join(cross_feat, how="left")

    feat["symbol"] = symbol
    feat["interval"] = interval
    return feat


def build_features_dataset(
    symbols: Sequence[str],
    intervals: Sequence[Interval],
    parquet_dir: str = "data/parquet",
    out_dir: str = "data/datasets",
) -> dict[str, list[str]]:
    written: dict[str, list[str]] = {}
    for interval in intervals:
        paths: list[str] = []
        for sym in symbols:
            df = build_features_for_symbol(sym, interval, parquet_dir)
            part_dir = os.path.join(out_dir, f"interval={interval}", f"symbol={sym}")
            os.makedirs(part_dir, exist_ok=True)
            path = os.path.join(part_dir, "data.parquet")
            if df.empty:
                # schema-preserving empty file
                df = pd.DataFrame(
                    columns=pd.Index([
                        "r_3","r_5","r_10","r_20","r_30","r_60","macd","macd_signal","macd_hist","adx_14","sharpe_20","mom_strength",
                        "rsi_2","rsi_14","ema20_z","%b","price_to_sma50","price_to_sma200","above_ema20",
                        "gap_z","range_intraday","hl_pct_ma","vol_z","vol_ratio","autocorr_1","autocorr_5",
                        "atr_14","vol_ratio_term","dist_from_high","dist_from_low",
                        "vix_level","vix_trend","bond_stock_corr","credit_spread","relative_strength","spy_trend","market_breadth",
                        "open","high","low","close","volume","next_open","next_close","symbol","interval"
                    ]),
                    index=pd.DatetimeIndex([], name="datetime"),
                )
            df.to_parquet(path, engine="pyarrow", compression="zstd")
            paths.append(path)
        written[interval] = paths
    return written


__all__ = [
    "FeatureSpec",
    "build_features_for_symbol",
    "build_features_dataset",
]