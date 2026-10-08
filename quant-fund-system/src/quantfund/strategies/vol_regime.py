"""
Volatility regime strategy: trade when ML predicts low volatility periods.

Logic:
- ML model predicts p(high_vol) for next 5 days
- When p(high_vol) < 0.3 → low vol regime expected → increase position sizes
- When p(high_vol) > 0.7 → high vol regime expected → reduce/exit positions
- In between: scale proportionally

This works because:
1. Mean reversion and momentum work better in low vol environments
2. Volatility is more predictable than direction (vol clusters)
3. Position sizing based on predicted vol improves risk-adjusted returns
"""
from __future__ import annotations

import pandas as pd


def vol_regime_scaler(proba_high_vol: pd.Series, boost_in_low_vol: bool = True) -> pd.Series:
    """
    Convert high vol probability to position size scaler.

    Args:
        proba_high_vol: Probability of high volatility regime (0-1)
        boost_in_low_vol: If True, boost positions in low vol; if False, just reduce in high vol

    Returns:
        Position size multiplier
    """
    # Invert: low vol prob = 1 - high vol prob
    p_low_vol = 1.0 - proba_high_vol

    if boost_in_low_vol:
        # Boost in confident low vol, reduce in confident high vol
        # p_low_vol > 0.7 → scale = 1.3 (boost 30%)
        # p_low_vol = 0.5 → scale = 1.0 (neutral)
        # p_low_vol < 0.3 → scale = 0.7 (reduce 30%)
        scaler = 1.0 + 0.6 * (p_low_vol - 0.5)  # ranges from 0.7 to 1.3
        scaler = scaler.clip(0.7, 1.3)
    else:
        # Conservative: only reduce in high vol, never boost
        # p_low_vol > 0.6 → scale = 1.0 (full size)
        # p_low_vol < 0.2 → scale = 0.5 (half size)
        scaler = 0.5 + 0.5 * ((p_low_vol - 0.2) / 0.4).clip(0.0, 1.0)

    return scaler


def vol_regime_signals(
    interval_df: pd.DataFrame,
    base_signal: pd.Series,
    proba_high_vol: pd.Series,
) -> pd.Series:
    """
    Modulate base strategy signals by predicted volatility regime.

    Args:
        interval_df: DataFrame with features
        base_signal: Base strategy signal (e.g., from momo/meanrev)
        proba_high_vol: ML probability of high volatility

    Returns:
        Adjusted signal scaled by vol regime
    """
    if interval_df.empty:
        return pd.Series(dtype=float, index=interval_df.index, name="vol_regime_signal")

    scaler = vol_regime_scaler(proba_high_vol)

    # Apply vol regime scaling to base signal
    signal = base_signal * scaler

    return signal.rename("vol_regime_signal")


__all__ = ["vol_regime_scaler", "vol_regime_signals"]
