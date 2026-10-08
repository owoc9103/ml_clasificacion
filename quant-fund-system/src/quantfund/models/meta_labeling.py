"""
Meta-Labeling for Bet Sizing

Based on:
- López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 3
- Meta-labeling: Using ML to learn how much to bet, not what to bet on

Meta-labeling is a two-stage approach:
1. Primary model: Predicts direction (long/short) - our existing model
2. Meta-model: Predicts bet size (0-100%) given primary prediction + market features

Key concept:
- Primary model has directional edge (e.g., 53% accuracy)
- Meta-model filters low-quality signals (improves precision)
- Result: Fewer trades, but higher win rate

Example:
- Primary says "Long" with 55% probability
- Meta-model sees: low volatility, strong trend, high volume
- Meta-model says: "Bet 75%" (high confidence)
- Final position: 0.75 * max_position

Benefits:
- Reduces false positives (fewer losing trades)
- Concentrates capital on high-quality signals
- Uses triple barrier correctly (bet sizing, not direction)
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from quantfund.models.triple_barrier import (
    triple_barrier_labels,
    TripleBarrierConfig,
)


@dataclass(frozen=True)
class MetaLabelConfig:
    """Configuration for meta-labeling."""
    # Triple barrier parameters for meta-labels
    profit_pct: float = 0.09
    stop_pct: float = 0.09
    max_holding_bars: int = 29

    # Meta-label thresholds
    # If barrier_return > profit_threshold → label=1 (good bet)
    # If barrier_return < stop_threshold → label=0 (bad bet)
    profit_threshold: float = 0.03  # 3% minimum profit to consider "good"
    stop_threshold: float = -0.03   # -3% stop to consider "bad"

    # Probability threshold for primary model
    # Only consider signals with primary_prob > min_primary_prob
    min_primary_prob: float = 0.52  # Require 52%+ confidence from primary


def generate_meta_labels(
    df: pd.DataFrame,
    primary_predictions: pd.Series,
    primary_side: pd.Series,
    config: Optional[MetaLabelConfig] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Generate meta-labels for bet sizing.

    Meta-labels indicate whether a primary model's prediction will be profitable.

    Process:
    1. Filter to only primary model's positive predictions (side == 1)
    2. Apply triple barrier to those predictions
    3. Label as 1 if profitable, 0 if not

    Args:
        df: DataFrame with OHLCV data
        primary_predictions: Primary model probabilities (0-1)
        primary_side: Primary model direction (1=long, 0=short/neutral)
        config: Meta-label configuration

    Returns:
        meta_labels: Binary labels (1=good bet, 0=bad bet)
        barrier_results: DataFrame with barrier outcomes for analysis

    Example:
        >>> meta_labels, barrier_df = generate_meta_labels(
        ...     df, primary_probs, primary_sides
        ... )
        >>> print(f"Good bets: {(meta_labels == 1).sum()}")
    """
    if config is None:
        config = MetaLabelConfig()

    # Only consider primary model's long predictions
    # Meta-labeling asks: "Given primary says long, should we bet?"
    primary_mask = (primary_side == 1) & (primary_predictions >= config.min_primary_prob)

    # Apply triple barrier to primary's predictions
    tb_config = TripleBarrierConfig(
        profit_pct=config.profit_pct,
        stop_pct=config.stop_pct,
        max_holding_bars=config.max_holding_bars,
        use_return_sign_on_timeout=False,  # Keep timeout as neutral
    )

    # Get barrier outcomes
    from quantfund.models.triple_barrier import apply_triple_barrier, get_barriers

    close = df['close']
    upper_barrier, lower_barrier = get_barriers(
        close, config.profit_pct, config.stop_pct
    )

    barrier_results = apply_triple_barrier(
        close,
        upper_barrier,
        lower_barrier,
        config.max_holding_bars,
    )

    # Meta-label: 1 if bet would be profitable, 0 otherwise
    # Profitable = hit upper barrier OR timeout with positive return > threshold
    meta_labels = pd.Series(np.nan, index=df.index)

    for idx in df.index:
        if not primary_mask.loc[idx]:
            # Not a primary prediction, no meta-label
            continue

        barrier = barrier_results.loc[idx, 'barrier_touched']
        ret = barrier_results.loc[idx, 'return']

        if pd.isna(barrier):
            # Not enough future data
            continue

        # Label as 1 (good bet) if:
        # - Hit upper barrier (profit take)
        # - OR timeout with return > profit_threshold
        if barrier == 1:  # Upper barrier
            meta_labels.loc[idx] = 1
        elif barrier == -1:  # Lower barrier (stop loss)
            meta_labels.loc[idx] = 0
        elif barrier == 0:  # Timeout
            # Use return to decide
            if ret >= config.profit_threshold:
                meta_labels.loc[idx] = 1
            elif ret <= config.stop_threshold:
                meta_labels.loc[idx] = 0
            else:
                # Neutral, don't include in training
                pass

    return meta_labels, barrier_results


def build_meta_features(
    df: pd.DataFrame,
    primary_predictions: pd.Series,
) -> pd.DataFrame:
    """
    Build features for meta-model.

    Meta-model features should capture:
    1. Primary model confidence (probability)
    2. Market regime (volatility, trend strength)
    3. Signal quality indicators

    Args:
        df: DataFrame with base features
        primary_predictions: Primary model probabilities

    Returns:
        DataFrame with meta-features

    Example:
        >>> meta_feat = build_meta_features(df, primary_probs)
        >>> print(meta_feat.columns)
    """
    meta = pd.DataFrame(index=df.index)

    # 1. Primary model features
    meta['primary_prob'] = primary_predictions
    meta['primary_confidence'] = np.abs(primary_predictions - 0.5) * 2  # 0-1 scale

    # 2. Volatility regime (lower vol = better for trend following)
    if 'atr_14' in df.columns:
        meta['atr_14'] = df['atr_14']
        meta['atr_percentile'] = df['atr_14'].rolling(252).rank(pct=True)

    # 3. Trend strength
    if 'adx_14' in df.columns:
        meta['adx_14'] = df['adx_14']  # Higher ADX = stronger trend

    if 'macd_hist' in df.columns:
        meta['macd_hist'] = df['macd_hist']
        meta['macd_hist_sign'] = np.sign(df['macd_hist'])

    # 4. Momentum strength (stronger momentum = better for continuation)
    if 'r_20' in df.columns:
        meta['r_20'] = df['r_20']
        meta['r_20_abs'] = np.abs(df['r_20'])

    # 5. Mean-reversion indicators (avoid overbought/oversold)
    if 'rsi_14' in df.columns:
        meta['rsi_14'] = df['rsi_14']
        meta['rsi_neutral'] = np.abs(df['rsi_14'] - 50)  # Distance from 50

    # 6. Volume confirmation
    if 'vol_ratio' in df.columns:
        meta['vol_ratio'] = df['vol_ratio']

    # 7. Recent performance (avoid after big moves)
    if 'r_5' in df.columns:
        meta['r_5'] = df['r_5']

    # 8. Volatility ratio (avoid high vol expansion)
    if 'vol_ratio_term' in df.columns:
        meta['vol_ratio_term'] = df['vol_ratio_term']

    # 9. Price position (better to enter mid-range than extremes)
    if 'dist_from_high' in df.columns and 'dist_from_low' in df.columns:
        meta['dist_from_high'] = df['dist_from_high']
        meta['dist_from_low'] = df['dist_from_low']

    # 10. Regime filter
    if 'spy_trend' in df.columns:
        meta['spy_trend'] = df['spy_trend']

    if 'vol_regime_ok' in df.columns:
        meta['vol_regime_ok'] = df['vol_regime_ok']

    # Fill NaN with 0 for meta-features
    meta = meta.fillna(0)

    return meta


def apply_meta_model(
    primary_predictions: pd.Series,
    primary_side: pd.Series,
    meta_predictions: pd.Series,
    meta_threshold: float = 0.5,
) -> pd.Series:
    """
    Combine primary and meta-model predictions.

    Logic:
    - If primary says long AND meta says good bet → bet 100%
    - If primary says long BUT meta says bad bet → bet 0% (filter out)
    - If primary says short/neutral → bet 0%

    Args:
        primary_predictions: Primary model probabilities
        primary_side: Primary model direction (1=long, 0=other)
        meta_predictions: Meta-model probabilities (quality of bet)
        meta_threshold: Threshold for meta-model (default 0.5)

    Returns:
        final_signals: Combined signals (0-1 scale)

    Example:
        >>> final = apply_meta_model(primary_probs, primary_sides, meta_probs)
        >>> print(f"Filtered out: {(final == 0).sum() - (primary_side == 0).sum()}")
    """
    # Start with primary predictions
    final_signals = primary_predictions.copy()

    # Filter: Only keep signals where meta says "good bet"
    # If meta_prob < threshold, set signal to 0 (don't bet)
    filter_mask = (primary_side == 1) & (meta_predictions < meta_threshold)
    final_signals.loc[filter_mask] = 0.0

    # Scale by meta confidence (optional enhancement)
    # Signals where meta is confident get full weight
    # Signals where meta is uncertain get reduced weight
    scale_mask = (primary_side == 1) & (meta_predictions >= meta_threshold)
    final_signals.loc[scale_mask] *= meta_predictions.loc[scale_mask]

    return final_signals


def evaluate_meta_labeling(
    y_true: pd.Series,
    primary_predictions: pd.Series,
    meta_predictions: pd.Series,
    meta_threshold: float = 0.5,
) -> dict:
    """
    Evaluate meta-labeling impact.

    Compares:
    - Primary only: All primary signals
    - Primary + Meta: Filtered signals

    Metrics:
    - Precision improvement (fewer false positives)
    - Recall reduction (trade-off: fewer trades)
    - F1 score (overall quality)

    Args:
        y_true: True labels
        primary_predictions: Primary model predictions
        meta_predictions: Meta-model predictions
        meta_threshold: Meta-model threshold

    Returns:
        dict with comparison metrics

    Example:
        >>> results = evaluate_meta_labeling(y_test, primary_probs, meta_probs)
        >>> print(f"Precision gain: {results['precision_gain']:.1%}")
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    # Convert probabilities to binary predictions
    primary_binary = (primary_predictions >= 0.5).astype(int)

    # Apply meta-filtering
    meta_filter = (meta_predictions >= meta_threshold)
    filtered_binary = primary_binary.copy()
    filtered_binary.loc[~meta_filter] = 0  # Filter out low-quality

    # Compute metrics for samples with predictions
    primary_mask = (primary_binary == 1)
    filtered_mask = (filtered_binary == 1)

    # Primary only
    if primary_mask.sum() > 0:
        primary_precision = precision_score(y_true[primary_mask], primary_binary[primary_mask])
        primary_recall = recall_score(y_true, primary_binary)
        primary_f1 = f1_score(y_true, primary_binary)
    else:
        primary_precision = primary_recall = primary_f1 = 0.0

    # Primary + Meta
    if filtered_mask.sum() > 0:
        filtered_precision = precision_score(y_true[filtered_mask], filtered_binary[filtered_mask])
        filtered_recall = recall_score(y_true, filtered_binary)
        filtered_f1 = f1_score(y_true, filtered_binary)
    else:
        filtered_precision = filtered_recall = filtered_f1 = 0.0

    # Compute gains
    precision_gain = (filtered_precision - primary_precision) / (primary_precision + 1e-10)

    return {
        'primary': {
            'precision': primary_precision,
            'recall': primary_recall,
            'f1': primary_f1,
            'n_trades': primary_mask.sum(),
        },
        'filtered': {
            'precision': filtered_precision,
            'recall': filtered_recall,
            'f1': filtered_f1,
            'n_trades': filtered_mask.sum(),
        },
        'gains': {
            'precision_gain': precision_gain,
            'recall_change': filtered_recall - primary_recall,
            'f1_change': filtered_f1 - primary_f1,
            'trades_reduction': 1 - (filtered_mask.sum() / (primary_mask.sum() + 1e-10)),
        }
    }


__all__ = [
    'MetaLabelConfig',
    'generate_meta_labels',
    'build_meta_features',
    'apply_meta_model',
    'evaluate_meta_labeling',
]
