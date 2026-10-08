"""
Fractional Differentiation

Based on:
- López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 5
- Addresses non-stationarity while preserving maximum memory

Fractional differentiation transforms a non-stationary series into stationary
while retaining more memory than integer differentiation (d=1).

Key concept:
- d=0: No differencing (non-stationary, 100% memory)
- d=0.5: Half differentiation (partially stationary, ~95% memory)
- d=1: Full differentiation (stationary, minimal memory)

The optimal d balances stationarity and memory retention.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.special import comb


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for fixed-window fractional differentiation (FFD).

    FFD uses a fixed-length window to approximate fractional differentiation,
    making it more practical for real-time applications than expanding window.

    The weights follow: w_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))

    Args:
        d: Fractional differentiation order (0 < d < 1)
        threshold: Drop weights below this value to limit window size

    Returns:
        Array of weights for FFD

    Example:
        >>> weights = get_weights_ffd(d=0.5)
        >>> len(weights)  # Typically 50-100 for d=0.5
    """
    if not (0 < d < 1):
        raise ValueError(f"d must be in (0, 1), got {d}")

    # Compute weights using binomial coefficients
    # w_k = (-1)^k * C(d, k) where C is binomial coefficient
    weights = [1.0]  # w_0 = 1
    k = 1

    while True:
        # Recursive formula: w_k = -w_{k-1} * (d - k + 1) / k
        w_new = -weights[-1] * (d - k + 1) / k

        if abs(w_new) < threshold:
            break

        weights.append(w_new)
        k += 1

        if k > 1000:  # Safety limit
            break

    return np.array(weights)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fixed-window fractional differentiation to a series.

    FFD computes: X_t^d = Σ(k=0 to K) w_k * X_{t-k}
    where w_k are the fractional weights and K is determined by threshold.

    Args:
        series: Time series to differentiate
        d: Fractional order (0 < d < 1)
        threshold: Weight threshold for window truncation

    Returns:
        Fractionally differentiated series with NaN for first K values

    Example:
        >>> prices = pd.Series([100, 101, 102, 101, 100])
        >>> frac_diff = frac_diff_ffd(prices, d=0.5)
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights) - 1

    if len(series) <= width:
        # Not enough data for differentiation
        return pd.Series(index=series.index, dtype=float)

    # Apply convolution
    # For each position t, compute weighted sum of X[t], X[t-1], ..., X[t-K]
    result = np.full(len(series), np.nan)

    for i in range(width, len(series)):
        # Get window: [i-width, i-width+1, ..., i]
        window = series.iloc[i-width:i+1].values
        # Reverse window to align with weights (newest first)
        window = window[::-1]
        # Apply weights
        result[i] = np.dot(weights, window)

    return pd.Series(result, index=series.index)


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, float, dict]:
    """
    Test stationarity using Augmented Dickey-Fuller (ADF) test.

    ADF tests null hypothesis: series has a unit root (non-stationary)
    If p-value < alpha, reject null → series is stationary

    Args:
        series: Time series to test
        alpha: Significance level (default 5%)

    Returns:
        is_stationary: True if series is stationary
        p_value: ADF test p-value
        results: Full ADF test results dict

    Example:
        >>> is_stat, pval, _ = test_stationarity(prices)
        >>> print(f"Stationary: {is_stat}, p-value: {pval:.4f}")
    """
    # Drop NaN values
    clean = series.dropna()

    if len(clean) < 20:
        # Not enough data for meaningful test
        return False, 1.0, {}

    # Run ADF test
    adf_result = adfuller(clean, autolag='AIC')

    p_value = adf_result[1]
    is_stationary = p_value < alpha

    results = {
        'adf_stat': adf_result[0],
        'p_value': p_value,
        'n_lags': adf_result[2],
        'n_obs': adf_result[3],
        'critical_values': adf_result[4],
    }

    return is_stationary, p_value, results


def compute_memory_retention(original: pd.Series, transformed: pd.Series) -> float:
    """
    Compute correlation between original and transformed series.

    Higher correlation = more memory retained from original series.
    López de Prado suggests retaining 90%+ correlation (0.90+).

    Args:
        original: Original series
        transformed: Fractionally differentiated series

    Returns:
        Correlation coefficient (Pearson)

    Example:
        >>> corr = compute_memory_retention(prices, frac_diff_prices)
        >>> print(f"Memory retention: {corr:.1%}")
    """
    # Align series (transformed has NaN at start)
    mask = transformed.notna() & original.notna()

    if mask.sum() < 10:
        return 0.0

    corr = original[mask].corr(transformed[mask])
    return corr if not np.isnan(corr) else 0.0


def find_min_d(series: pd.Series,
               d_range: Tuple[float, float] = (0.0, 1.0),
               step: float = 0.05,
               target_pvalue: float = 0.05,
               min_memory: float = 0.90) -> Tuple[float, dict]:
    """
    Find minimum d that achieves stationarity while retaining sufficient memory.

    Strategy:
    1. Start with d=0 (no differencing)
    2. Incrementally increase d
    3. Test stationarity at each step
    4. Stop when series becomes stationary AND memory > min_memory
    5. Return smallest d meeting criteria

    Args:
        series: Time series to analyze
        d_range: Range of d values to search
        step: Increment for d search
        target_pvalue: ADF p-value threshold for stationarity
        min_memory: Minimum correlation to retain

    Returns:
        optimal_d: Minimum d achieving stationarity
        results: Dict with stationarity and memory metrics for each d

    Example:
        >>> optimal_d, results = find_min_d(prices)
        >>> print(f"Optimal d: {optimal_d:.2f}")
    """
    d_min, d_max = d_range
    d_values = np.arange(d_min + step, d_max, step)  # Start from step, not 0

    results = []

    for d in d_values:
        # Apply fractional differentiation
        frac_series = frac_diff_ffd(series, d)

        # Test stationarity
        is_stat, pval, adf_results = test_stationarity(frac_series)

        # Compute memory retention
        memory = compute_memory_retention(series, frac_series)

        results.append({
            'd': d,
            'is_stationary': is_stat,
            'p_value': pval,
            'memory': memory,
            'adf_stat': adf_results.get('adf_stat', np.nan),
        })

        # Stop if we found stationary series with sufficient memory
        if is_stat and memory >= min_memory:
            return d, {'search_results': pd.DataFrame(results)}

    # If no d meets criteria, return highest d tested
    if results:
        # Return d with best stationarity (lowest p-value)
        df_results = pd.DataFrame(results)
        best_idx = df_results['p_value'].idxmin()
        return df_results.loc[best_idx, 'd'], {'search_results': df_results}

    return d_max, {'search_results': pd.DataFrame()}


def add_fracdiff_features(df: pd.DataFrame,
                          price_col: str = 'close',
                          d_values: list = [0.3, 0.4, 0.5]) -> pd.DataFrame:
    """
    Add fractionally differentiated features to a dataframe.

    Creates new columns for each d value:
    - fracdiff_d30: d=0.3 (light differencing, max memory)
    - fracdiff_d40: d=0.4 (moderate differencing)
    - fracdiff_d50: d=0.5 (heavy differencing, more stationary)

    Args:
        df: DataFrame with price column
        price_col: Name of price column to differentiate
        d_values: List of d values to compute

    Returns:
        DataFrame with added fracdiff_dXX columns

    Example:
        >>> df = add_fracdiff_features(df, 'close', [0.3, 0.5])
        >>> df[['close', 'fracdiff_d30', 'fracdiff_d50']].head()
    """
    df = df.copy()

    for d in d_values:
        col_name = f'fracdiff_d{int(d*100):02d}'
        df[col_name] = frac_diff_ffd(df[price_col], d)

    return df


__all__ = [
    'get_weights_ffd',
    'frac_diff_ffd',
    'test_stationarity',
    'compute_memory_retention',
    'find_min_d',
    'add_fracdiff_features',
]
