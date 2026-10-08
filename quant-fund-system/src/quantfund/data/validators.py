"""
Data validation utilities for price and feature data.

Validates data quality before use in strategies and backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quantfund.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.is_valid


def validate_ohlcv(df: pd.DataFrame, symbol: str = "unknown") -> ValidationResult:
    """Validate OHLCV price data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        symbol: Symbol name for logging

    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []

    # Check required columns
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if df.empty:
        warnings.append(f"{symbol}: Empty DataFrame")
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    # Check for NaN values
    for col in required:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            errors.append(f"{symbol}: {nan_count} NaN values in {col}")

    # Check price relationships: high >= low, high >= open, high >= close, low <= open, low <= close
    if not (df["high"] >= df["low"]).all():
        errors.append(f"{symbol}: Found bars where high < low")

    if not (df["high"] >= df["open"]).all():
        errors.append(f"{symbol}: Found bars where high < open")

    if not (df["high"] >= df["close"]).all():
        errors.append(f"{symbol}: Found bars where high < close")

    if not (df["low"] <= df["open"]).all():
        errors.append(f"{symbol}: Found bars where low > open")

    if not (df["low"] <= df["close"]).all():
        errors.append(f"{symbol}: Found bars where low > close")

    # Check for negative prices
    for col in required:
        if (df[col] <= 0).any():
            errors.append(f"{symbol}: Found negative or zero prices in {col}")

    # Check for extreme price moves (>50% single bar, likely split/error)
    returns = df["close"].pct_change().abs()
    extreme_moves = returns > 0.5
    if extreme_moves.any():
        count = extreme_moves.sum()
        warnings.append(f"{symbol}: {count} bars with >50% price change (possible splits/errors)")

    # Check for zero volume
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            warnings.append(f"{symbol}: {zero_vol} bars with zero volume (halted/illiquid)")

    # Check for data staleness (if index is DatetimeIndex)
    if isinstance(df.index, pd.DatetimeIndex):
        gaps = df.index.to_series().diff()
        # Expected frequency (assume most common gap is the intended frequency)
        mode_gap = gaps.mode()[0] if len(gaps.mode()) > 0 else pd.Timedelta(days=1)
        large_gaps = gaps > mode_gap * 5  # Flag gaps >5x expected
        if large_gaps.any():
            count = large_gaps.sum()
            warnings.append(f"{symbol}: {count} large data gaps (>5x expected frequency)")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_features(df: pd.DataFrame, symbol: str = "unknown") -> ValidationResult:
    """Validate feature data.

    Args:
        df: DataFrame with technical indicators
        symbol: Symbol name for logging

    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []

    if df.empty:
        warnings.append(f"{symbol}: Empty feature DataFrame")
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    # Check for inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            errors.append(f"{symbol}: {inf_count} inf values in {col}")

    # Check for excessive NaN (>50% of rows)
    for col in numeric_cols:
        nan_pct = df[col].isna().sum() / len(df)
        if nan_pct > 0.5:
            warnings.append(f"{symbol}: {col} has {nan_pct:.1%} NaN values")

    # Check index is sorted
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            errors.append(f"{symbol}: Index is not sorted")

    # Check for duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        errors.append(f"{symbol}: {dup_count} duplicate timestamps")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_signals(signals: pd.Series, symbol: str = "unknown") -> ValidationResult:
    """Validate trading signals.

    Args:
        signals: Series with signal values (should be in [0, 1])
        symbol: Symbol name for logging

    Returns:
        ValidationResult with errors and warnings
    """
    errors = []
    warnings = []

    if signals.empty:
        warnings.append(f"{symbol}: Empty signal series")
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings)

    # Check range [0, 1]
    if (signals < 0).any():
        count = (signals < 0).sum()
        errors.append(f"{symbol}: {count} signals < 0")

    if (signals > 1).any():
        count = (signals > 1).sum()
        errors.append(f"{symbol}: {count} signals > 1")

    # Check for NaN
    nan_count = signals.isna().sum()
    if nan_count > 0:
        errors.append(f"{symbol}: {nan_count} NaN signals")

    # Check for inf
    inf_count = np.isinf(signals).sum()
    if inf_count > 0:
        errors.append(f"{symbol}: {inf_count} inf signals")

    # Warn if all zeros (no trading)
    if (signals == 0).all():
        warnings.append(f"{symbol}: All signals are zero (no trading)")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


__all__ = [
    "ValidationResult",
    "validate_ohlcv",
    "validate_features",
    "validate_signals",
]
