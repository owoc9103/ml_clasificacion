"""
Advanced labeling strategies for ML models.

Supports multiple prediction targets:
1. Binary direction (baseline): predict if next return > cost
2. Volatility regime: predict high/low volatility periods
3. Trend strength: predict strong trend vs choppy/ranging
4. Risk-adjusted returns: predict positive Sharpe over next N days
"""
from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd

LabelType = Literal["binary", "vol_regime", "trend_strength", "sharpe_regime"]


def label_binary_direction(df: pd.DataFrame, cost: float = 0.0002) -> pd.Series:
    """
    Binary classification: next return > cost

    Current baseline approach.
    """
    ret_next = (df["next_close"] / df["next_open"]) - 1.0
    return (ret_next - cost > 0).astype(int)


def label_volatility_regime(df: pd.DataFrame, lookforward: int = 5) -> pd.Series:
    """
    Predict if next N days will be high volatility.

    High vol = realized vol over next N days > median vol
    This is more predictable than direction because vol clusters.
    """
    # Compute forward-looking realized vol
    close = df["close"]
    ret = close.pct_change()

    # Forward volatility (next N days)
    fwd_vol = ret.shift(-1).rolling(lookforward).std().shift(-lookforward + 1)

    # Median volatility over full sample
    median_vol = fwd_vol.median()

    # Label: 1 if high vol, 0 if low vol
    y = (fwd_vol > median_vol).astype(int)

    return y


def label_trend_strength(df: pd.DataFrame, lookforward: int = 10) -> pd.Series:
    """
    Predict if next N days will have strong trend.

    Strong trend = price moves consistently in one direction
    Measured by: abs(return) > volatility (high signal/noise ratio)
    """
    close = df["close"]

    # Forward return and volatility over next N days
    fwd_ret = (close.shift(-lookforward) / close) - 1.0
    fwd_vol = close.pct_change().shift(-1).rolling(lookforward).std().shift(-lookforward + 1)

    # Trend strength = return magnitude relative to volatility
    # Strong trend if abs(return) > 1.5 * volatility (good Sharpe-like signal)
    trend_signal = np.abs(fwd_ret) / (fwd_vol * np.sqrt(lookforward))

    # Label: 1 if strong trend (signal > 1.5), 0 if choppy
    y = (trend_signal > 1.5).astype(int)

    return y


def label_sharpe_regime(df: pd.DataFrame, lookforward: int = 20) -> pd.Series:
    """
    Predict if next N days will have positive risk-adjusted returns.

    Positive Sharpe = (mean return / std return) > 0 over next N days
    This is the most actionable target - we want to trade when Sharpe is positive.
    """
    close = df["close"]
    ret = close.pct_change()

    # Forward Sharpe over next N days
    fwd_mean = ret.shift(-1).rolling(lookforward).mean().shift(-lookforward + 1)
    fwd_std = ret.shift(-1).rolling(lookforward).std().shift(-lookforward + 1)
    fwd_sharpe = fwd_mean / fwd_std.replace(0, np.nan)

    # Label: 1 if positive Sharpe, 0 otherwise
    y = (fwd_sharpe > 0).astype(int)

    return y


def create_labels(
    df: pd.DataFrame,
    label_type: LabelType = "binary",
    cost: float = 0.0002,
    lookforward: int = 10,
) -> pd.Series:
    """
    Create labels based on specified strategy.

    Args:
        df: DataFrame with OHLC data (must have 'close', 'next_open', 'next_close')
        label_type: Type of label to create
        cost: Transaction cost (for binary direction)
        lookforward: Days to look forward (for regime predictions)

    Returns:
        Series of binary labels (0 or 1)
    """
    if label_type == "binary":
        return label_binary_direction(df, cost)
    elif label_type == "vol_regime":
        return label_volatility_regime(df, lookforward)
    elif label_type == "trend_strength":
        return label_trend_strength(df, lookforward)
    elif label_type == "sharpe_regime":
        return label_sharpe_regime(df, lookforward)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


__all__ = [
    "LabelType",
    "create_labels",
    "label_binary_direction",
    "label_volatility_regime",
    "label_trend_strength",
    "label_sharpe_regime",
]
