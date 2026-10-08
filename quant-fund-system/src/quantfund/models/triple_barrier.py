"""
Triple Barrier Labeling Method

Based on:
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Enhanced GA-driven method (MDPI Mathematics, 2024)
- Korean market optimization study (arXiv 2024)

The triple barrier method labels samples based on which barrier is touched first:
1. Upper barrier (profit take): price * (1 + profit_pct)
2. Lower barrier (stop loss): price * (1 - stop_pct)
3. Vertical barrier (time limit): max_holding_period bars

This captures magnitude, timing, and risk-reward, unlike simple binary returns.
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    profit_pct: float = 0.09  # 9% from Korean market study (2024)
    stop_pct: float = 0.09    # Symmetric barriers
    max_holding_bars: int = 29  # 29 days from research

    # Label encoding
    label_upper: int = 1   # Profit barrier hit (buy signal)
    label_lower: int = -1  # Stop barrier hit (sell signal)
    label_time: int = 0    # Time barrier hit (no action or sign-based)
    use_return_sign_on_timeout: bool = True  # If timeout, use return sign

    def __post_init__(self):
        """Validate parameters."""
        if self.profit_pct <= 0:
            raise ValueError("profit_pct must be > 0")
        if self.stop_pct <= 0:
            raise ValueError("stop_pct must be > 0")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")


def get_barriers(
    close: pd.Series,
    profit_pct: float,
    stop_pct: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate upper and lower price barriers.

    Args:
        close: Close prices
        profit_pct: Profit take percentage (e.g., 0.09 for 9%)
        stop_pct: Stop loss percentage (e.g., 0.09 for 9%)

    Returns:
        upper_barrier: Profit take prices
        lower_barrier: Stop loss prices
    """
    upper_barrier = close * (1.0 + profit_pct)
    lower_barrier = close * (1.0 - stop_pct)
    return upper_barrier, lower_barrier


def apply_triple_barrier(
    close: pd.Series,
    upper_barrier: pd.Series,
    lower_barrier: pd.Series,
    max_holding_bars: int,
) -> pd.DataFrame:
    """
    Apply triple barrier method to determine labels.

    For each bar, check which barrier is touched first within max_holding_bars:
    - Upper barrier → profit (label = 1)
    - Lower barrier → loss (label = -1)
    - Time barrier → timeout (label = 0 or based on return)

    Args:
        close: Close prices
        upper_barrier: Upper barrier prices
        lower_barrier: Lower barrier prices
        max_holding_bars: Maximum holding period

    Returns:
        DataFrame with columns:
        - barrier_touched: Which barrier was hit first (1=upper, -1=lower, 0=time)
        - touch_time: How many bars until barrier touch
        - return: Actual return at barrier touch
    """
    n = len(close)
    results = []

    for i in range(n):
        if i + max_holding_bars >= n:
            # Not enough future data
            results.append({
                'barrier_touched': np.nan,
                'touch_time': np.nan,
                'return': np.nan,
            })
            continue

        # Get current barriers
        upper = upper_barrier.iloc[i]
        lower = lower_barrier.iloc[i]
        entry_price = close.iloc[i]

        # Check future prices up to max_holding_bars
        future_prices = close.iloc[i+1:i+1+max_holding_bars]

        # Find first touch of each barrier
        upper_touch = (future_prices >= upper)
        lower_touch = (future_prices <= lower)

        upper_idx = upper_touch.idxmax() if upper_touch.any() else None
        lower_idx = lower_touch.idxmax() if lower_touch.any() else None

        # Determine which was touched first
        if upper_idx is not None and lower_idx is not None:
            # Both touched - which came first?
            upper_pos = future_prices.index.get_loc(upper_idx)
            lower_pos = future_prices.index.get_loc(lower_idx)

            if upper_pos < lower_pos:
                barrier = 1
                touch_time = upper_pos + 1
                exit_price = future_prices.iloc[upper_pos]
            else:
                barrier = -1
                touch_time = lower_pos + 1
                exit_price = future_prices.iloc[lower_pos]
        elif upper_idx is not None:
            # Only upper touched
            barrier = 1
            touch_time = future_prices.index.get_loc(upper_idx) + 1
            exit_price = future_prices.iloc[touch_time - 1]
        elif lower_idx is not None:
            # Only lower touched
            barrier = -1
            touch_time = future_prices.index.get_loc(lower_idx) + 1
            exit_price = future_prices.iloc[touch_time - 1]
        else:
            # Time barrier hit (timeout)
            barrier = 0
            touch_time = max_holding_bars
            exit_price = future_prices.iloc[-1]

        # Calculate actual return
        ret = (exit_price / entry_price) - 1.0

        results.append({
            'barrier_touched': barrier,
            'touch_time': touch_time,
            'return': ret,
        })

    return pd.DataFrame(results, index=close.index)


def triple_barrier_labels(
    df: pd.DataFrame,
    config: Optional[TripleBarrierConfig] = None,
) -> pd.Series:
    """
    Generate triple barrier labels for a dataset.

    Args:
        df: DataFrame with 'close' column
        config: Triple barrier configuration (uses defaults if None)

    Returns:
        Series of labels: 1 (long), -1 (short), 0 (neutral)
    """
    if config is None:
        config = TripleBarrierConfig()

    close = df['close'].astype(float)

    # Calculate barriers
    upper_barrier, lower_barrier = get_barriers(
        close,
        config.profit_pct,
        config.stop_pct,
    )

    # Apply triple barrier
    barrier_df = apply_triple_barrier(
        close,
        upper_barrier,
        lower_barrier,
        config.max_holding_bars,
    )

    # Convert to labels
    labels = barrier_df['barrier_touched'].copy()

    # Handle timeouts
    if config.use_return_sign_on_timeout:
        timeout_mask = (labels == 0)
        timeout_returns = barrier_df.loc[timeout_mask, 'return']
        labels.loc[timeout_mask] = np.sign(timeout_returns)

    # Map to label encoding
    label_map = {
        1: config.label_upper,   # Profit barrier
        -1: config.label_lower,  # Stop barrier
        0: config.label_time,    # Time barrier
    }
    labels = labels.map(label_map)

    return labels


def optimize_barrier_parameters(
    df: pd.DataFrame,
    profit_range: Tuple[float, float] = (0.03, 0.15),
    stop_range: Tuple[float, float] = (0.03, 0.15),
    holding_range: Tuple[int, int] = (5, 50),
    n_samples: int = 20,
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    Grid search to find optimal triple barrier parameters.

    Optimizes for label balance and coverage (not NaN rate).

    Args:
        df: DataFrame with 'close' column
        profit_range: (min, max) profit percentage
        stop_range: (min, max) stop percentage
        holding_range: (min, max) holding bars
        n_samples: Number of samples per dimension

    Returns:
        best_profit_pct: Optimal profit percentage
        best_stop_pct: Optimal stop percentage
        best_holding: Optimal holding bars
        results_df: DataFrame with all tested combinations and metrics
    """
    # Generate parameter grid
    profit_vals = np.linspace(profit_range[0], profit_range[1], n_samples)
    stop_vals = np.linspace(stop_range[0], stop_range[1], n_samples)
    holding_vals = np.linspace(holding_range[0], holding_range[1], n_samples, dtype=int)

    results = []

    # Grid search
    for profit_pct in profit_vals:
        for stop_pct in stop_vals:
            for holding in holding_vals:
                config = TripleBarrierConfig(
                    profit_pct=profit_pct,
                    stop_pct=stop_pct,
                    max_holding_bars=holding,
                    use_return_sign_on_timeout=True,
                )

                labels = triple_barrier_labels(df, config)

                # Calculate metrics
                valid_labels = labels.dropna()
                if len(valid_labels) == 0:
                    continue

                # Label distribution
                n_upper = (valid_labels == 1).sum()
                n_lower = (valid_labels == -1).sum()
                n_neutral = (valid_labels == 0).sum()
                n_total = len(valid_labels)

                # Balance metric (closer to 0.33/0.33/0.33 is better)
                pct_upper = n_upper / n_total
                pct_lower = n_lower / n_total
                pct_neutral = n_neutral / n_total

                # Coverage (non-NaN rate)
                coverage = len(valid_labels) / len(labels)

                # Balance score: penalize imbalance (deviation from equal distribution)
                target = 1/3
                balance_score = 1.0 - (
                    abs(pct_upper - target) +
                    abs(pct_lower - target) +
                    abs(pct_neutral - target)
                ) / 2.0

                # Combined score (balance * coverage)
                score = balance_score * coverage

                results.append({
                    'profit_pct': profit_pct,
                    'stop_pct': stop_pct,
                    'holding_bars': holding,
                    'n_upper': n_upper,
                    'n_lower': n_lower,
                    'n_neutral': n_neutral,
                    'pct_upper': pct_upper,
                    'pct_lower': pct_lower,
                    'pct_neutral': pct_neutral,
                    'coverage': coverage,
                    'balance_score': balance_score,
                    'score': score,
                })

    results_df = pd.DataFrame(results).sort_values('score', ascending=False)

    # Best parameters
    best = results_df.iloc[0]
    best_profit_pct = best['profit_pct']
    best_stop_pct = best['stop_pct']
    best_holding = int(best['holding_bars'])

    return best_profit_pct, best_stop_pct, best_holding, results_df


def compare_label_methods(df: pd.DataFrame, cost: float = 0.0002) -> pd.DataFrame:
    """
    Compare binary labels vs triple barrier labels.

    Args:
        df: DataFrame with 'close', 'next_open', 'next_close'
        cost: Transaction cost for binary method

    Returns:
        DataFrame with label statistics for comparison
    """
    from quantfund.models.labels import label_binary_direction

    # Binary labels
    binary_labels = label_binary_direction(df, cost)
    binary_valid = binary_labels.dropna()

    # Triple barrier labels (default config)
    tb_config = TripleBarrierConfig()
    tb_labels = triple_barrier_labels(df, tb_config)
    tb_valid = tb_labels.dropna()

    # Statistics
    stats = {
        'method': ['Binary', 'Triple Barrier'],
        'n_samples': [len(binary_valid), len(tb_valid)],
        'coverage': [len(binary_valid) / len(df), len(tb_valid) / len(df)],
        'n_positive': [(binary_valid == 1).sum(), (tb_valid == 1).sum()],
        'n_negative': [(binary_valid == 0).sum(), (tb_valid == -1).sum()],
        'n_neutral': [0, (tb_valid == 0).sum()],
        'pct_positive': [(binary_valid == 1).mean(), (tb_valid == 1).mean()],
        'pct_negative': [(binary_valid == 0).mean(), (tb_valid == -1).mean()],
        'pct_neutral': [0, (tb_valid == 0).mean()],
    }

    return pd.DataFrame(stats)


__all__ = [
    'TripleBarrierConfig',
    'triple_barrier_labels',
    'get_barriers',
    'apply_triple_barrier',
    'optimize_barrier_parameters',
    'compare_label_methods',
]
