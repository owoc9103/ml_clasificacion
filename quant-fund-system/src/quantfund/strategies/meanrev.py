"""
Mean-reversion strategy: contrarian signals for oversold conditions.

Components:
- Oversold indicators: RSI2 < 10, RSI14 < 30, Bollinger %B < 0.2
- Price position: below EMA20, negative z-score
- Confirmation: not in downtrend (price near 52w low), decent volume
- Only trade mean reversion in established uptrends (above SMA200)

Signal is weighted combination, range [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def meanrev_signals(interval_df: pd.DataFrame, regime_ok: pd.Series) -> pd.Series:
    if interval_df.empty:
        return pd.Series(dtype=float, index=interval_df.index, name="meanrev_signal")

    # Oversold indicators (higher score = more oversold)
    rsi_2 = interval_df.get("rsi_2", pd.Series(50.0, index=interval_df.index))
    rsi_14 = interval_df.get("rsi_14", pd.Series(50.0, index=interval_df.index))
    percent_b = interval_df.get("%b", pd.Series(0.5, index=interval_df.index))

    # RSI oversold scores (normalize to [0, 1])
    rsi2_oversold = (10 - rsi_2.clip(0, 10)) / 10.0  # RSI2 < 10 → score = 1
    rsi14_oversold = (30 - rsi_14.clip(0, 30)) / 30.0  # RSI14 < 30 → score = 1
    bb_oversold = (0.2 - percent_b.clip(0, 0.2)) / 0.2  # %B < 0.2 → score = 1

    oversold_score = (
        rsi2_oversold * 0.4 +    # 40% - RSI2 is most sensitive
        rsi14_oversold * 0.35 +  # 35% - RSI14 confirmation
        bb_oversold * 0.25       # 25% - Bollinger Band oversold
    )

    # Price position (more negative z-score = more oversold)
    ema20_z = interval_df.get("ema20_z", pd.Series(0.0, index=interval_df.index))
    zscore_oversold = (-ema20_z.clip(-3, 0) / 3.0)  # Negative z → score > 0

    # Trend filter: only mean revert in uptrends (above SMA200)
    price_to_sma200 = interval_df.get("price_to_sma200", pd.Series(0.0, index=interval_df.index))
    uptrend_filter = (price_to_sma200 > -0.05).astype(float)  # Within 5% of SMA200

    # Avoid catching falling knives: not near 52-week low
    dist_from_low = interval_df.get("dist_from_low", pd.Series(0.0, index=interval_df.index))
    not_falling_knife = (dist_from_low > -0.15).astype(float)  # Not within 15% of 52w low

    # Volume confirmation (mean reversion works better with volume)
    vol_ratio = interval_df.get("vol_ratio", pd.Series(1.0, index=interval_df.index))
    vol_confirm = (vol_ratio > 0.8).astype(float) * 0.5 + 0.5  # scale to [0.5, 1.0]

    # Combine signals
    signal = (
        oversold_score * 0.40 +     # 40% - oversold indicators
        zscore_oversold * 0.30 +    # 30% - z-score position
        uptrend_filter * 0.15 +     # 15% - in uptrend
        not_falling_knife * 0.15    # 15% - not catching knife
    ) * vol_confirm                 # Multiply by volume confirmation

    # Clip to [0, 1] and apply regime filter
    signal = signal.clip(0.0, 1.0).fillna(0.0)
    signal = signal.where(regime_ok.reindex(interval_df.index).fillna(False), 0.0)

    return signal.rename("meanrev_signal")


__all__ = ["meanrev_signals"]