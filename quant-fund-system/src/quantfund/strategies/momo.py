"""
Momentum strategy: pure technical signals based on trend, momentum, and volume.

Components:
- Trend: price above moving averages (EMA20, SMA50, SMA200)
- Momentum: recent returns (r_5, r_10, r_20), MACD, ADX
- Volume: above-average volume confirms moves
- Strength: only enter on strong momentum (ADX > 25, positive MACD)

Signal is weighted combination, range [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momo_signals(interval_df: pd.DataFrame, proba: pd.Series, regime_ok: pd.Series) -> pd.Series:
    if interval_df.empty:
        return pd.Series(dtype=float, index=interval_df.index, name="momo_signal")

    # Trend components (above moving averages = bullish)
    above_ema20 = interval_df.get("above_ema20", pd.Series(0.5, index=interval_df.index))
    price_to_sma50 = interval_df.get("price_to_sma50", pd.Series(0.0, index=interval_df.index))
    price_to_sma200 = interval_df.get("price_to_sma200", pd.Series(0.0, index=interval_df.index))

    # Trend score: 1 if above all MAs, 0 if below all
    trend_score = (
        above_ema20 * 0.4 +  # 40% weight - short-term trend strength
        (price_to_sma50 > 0).astype(float) * 0.3 +  # 30% weight - medium-term
        (price_to_sma200 > 0).astype(float) * 0.3   # 30% weight - long-term
    )

    # Momentum components (positive returns = bullish)
    r_5 = interval_df.get("r_5", pd.Series(0.0, index=interval_df.index))
    r_10 = interval_df.get("r_10", pd.Series(0.0, index=interval_df.index))
    r_20 = interval_df.get("r_20", pd.Series(0.0, index=interval_df.index))

    # Momentum score: normalize returns to [0,1] using sigmoid-like scaling
    # Positive returns → score > 0.5, negative → score < 0.5
    momo_score = (
        (r_5 > 0).astype(float) * 0.4 +
        (r_10 > 0).astype(float) * 0.3 +
        (r_20 > 0).astype(float) * 0.3
    )

    # MACD confirmation (bullish when MACD > signal and histogram positive)
    macd = interval_df.get("macd", pd.Series(0.0, index=interval_df.index))
    macd_signal = interval_df.get("macd_signal", pd.Series(0.0, index=interval_df.index))
    macd_hist = interval_df.get("macd_hist", pd.Series(0.0, index=interval_df.index))

    macd_bullish = ((macd > macd_signal) & (macd_hist > 0)).astype(float)

    # ADX strength filter (only trade when trend is strong, ADX > 25)
    adx = interval_df.get("adx_14", pd.Series(20.0, index=interval_df.index))
    strength_filter = (adx > 25).astype(float)

    # Volume confirmation (above average volume)
    vol_ratio = interval_df.get("vol_ratio", pd.Series(1.0, index=interval_df.index))
    vol_confirm = (vol_ratio > 1.0).astype(float) * 0.5 + 0.5  # scale to [0.5, 1.0]

    # Combine signals with weights
    signal = (
        trend_score * 0.35 +       # 35% - trend is most important
        momo_score * 0.30 +        # 30% - momentum confirmation
        macd_bullish * 0.20 +      # 20% - MACD confirmation
        strength_filter * 0.15     # 15% - ADX strength
    ) * vol_confirm                # Multiply by volume confirmation

    # Clip to [0, 1] and apply regime filter
    signal = signal.clip(0.0, 1.0).fillna(0.0)
    signal = signal.where(regime_ok.reindex(interval_df.index).fillna(False), 0.0)

    return signal.rename("momo_signal")


__all__ = ["momo_signals"]