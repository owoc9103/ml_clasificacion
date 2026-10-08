#!/usr/bin/env python3
"""
Evaluate meta-labeling approach on test set.

Compares:
1. Primary model only (baseline)
2. Primary + Meta-model (filtered signals)

Usage:
    python scripts/evaluate_meta_model.py --start 2024-01-31 --end 2025-10-31 --interval 1d
"""
import click
import json
from pathlib import Path
from datetime import date

from quantfund.models.train import _collect_parquets, _concat_datasets, BASE_DROP
from quantfund.models.meta_labeling import build_meta_features, apply_meta_model, evaluate_meta_labeling
from quantfund.models.labels import label_binary_direction
from quantfund.utils.log import get_logger

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logger = get_logger(__name__)


@click.command()
@click.option("--start", required=True, type=str, help="Test start date (YYYY-MM-DD)")
@click.option("--end", required=True, type=str, help="Test end date (YYYY-MM-DD)")
@click.option("--interval", default="1d", type=click.Choice(["1d", "5d", "60m", "120m"]), show_default=True)
@click.option("--meta-threshold", default=0.5, type=float, show_default=True, help="Meta-model threshold")
def main(start, end, interval, meta_threshold):
    """Evaluate meta-labeling on test set."""
    logger.info(f"Evaluating meta-model: interval={interval}, period={start} to {end}")

    # Load data
    parts = _collect_parquets(interval)
    df = _concat_datasets(parts)

    # Filter by date
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            df = df.copy()
            df["timestamp"] = idx

    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = (pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")) + pd.Timedelta(days=1)
    m = (ts >= start_ts) & (ts <= end_ts)
    df = df.loc[m].copy()

    logger.info(f"Loaded {len(df)} test samples")

    # Generate true labels
    from quantfund.models.train import load_train_config
    config = load_train_config()
    y_true = label_binary_direction(df, config.cost_bps / 10000.0)

    # Load primary model
    primary_model_path = Path(f"artifacts/{interval}/model_all.joblib")
    primary_model = load(primary_model_path)
    if isinstance(primary_model, dict):
        primary_model = primary_model['model']

    logger.info("Generating primary model predictions...")
    drop_cols = set(BASE_DROP)
    numeric_cols = [c for c in df.columns if c not in drop_cols and c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].astype(np.float32)

    primary_probs = primary_model.predict_proba(X.values)[:, 1]
    primary_probs = pd.Series(primary_probs, index=df.index)
    primary_side = (primary_probs >= 0.5).astype(int)

    # Load meta-model
    meta_model_path = Path(f"artifacts/{interval}_meta/meta_model.joblib")
    if not meta_model_path.exists():
        logger.error(f"Meta-model not found: {meta_model_path}")
        logger.error("Please train meta-model first using: python scripts/train_meta_model.py")
        return

    meta_model_data = load(meta_model_path)
    meta_model = meta_model_data['model']
    meta_calibrator = meta_model_data.get('calibrator')

    logger.info("Building meta-features...")
    meta_features = build_meta_features(df, primary_probs)

    logger.info("Generating meta-model predictions...")
    meta_probs = meta_model.predict_proba(meta_features.values)[:, 1]
    if meta_calibrator is not None:
        meta_probs = meta_calibrator.transform(meta_probs)
    meta_probs = pd.Series(meta_probs, index=df.index)

    # Apply meta-model filtering
    logger.info(f"Applying meta-model with threshold={meta_threshold}...")
    final_signals = apply_meta_model(primary_probs, primary_side, meta_probs, meta_threshold)

    # Evaluate
    logger.info("Computing metrics...")

    # Align with valid labels
    valid_mask = y_true.notna()
    y_true_clean = y_true[valid_mask].astype(int)
    primary_probs_clean = primary_probs[valid_mask]
    meta_probs_clean = meta_probs[valid_mask]
    final_signals_clean = final_signals[valid_mask]
    primary_side_clean = primary_side[valid_mask]

    # Primary only metrics
    primary_auc = roc_auc_score(y_true_clean, primary_probs_clean)
    primary_binary = (primary_probs_clean >= 0.5).astype(int)
    primary_precision = precision_score(y_true_clean, primary_binary)
    primary_recall = recall_score(y_true_clean, primary_binary)
    primary_f1 = f1_score(y_true_clean, primary_binary)
    primary_n_trades = (primary_binary == 1).sum()

    # Meta-filtered metrics
    meta_binary = (final_signals_clean >= 0.5).astype(int)
    meta_auc = roc_auc_score(y_true_clean, final_signals_clean)
    meta_precision = precision_score(y_true_clean, meta_binary)
    meta_recall = recall_score(y_true_clean, meta_binary)
    meta_f1 = f1_score(y_true_clean, meta_binary)
    meta_n_trades = (meta_binary == 1).sum()

    # Compute improvements
    precision_gain = (meta_precision - primary_precision) / primary_precision
    recall_change = meta_recall - primary_recall
    f1_change = meta_f1 - primary_f1
    trades_reduction = 1 - (meta_n_trades / primary_n_trades)

    # Print results
    print("="*80)
    print("META-LABELING EVALUATION")
    print("="*80)
    print()
    print(f"Test Period: {start} to {end}")
    print(f"Samples: {len(y_true_clean)}")
    print()
    print("PRIMARY MODEL ONLY:")
    print(f"  ROC-AUC:     {primary_auc:.4f}")
    print(f"  Precision:   {primary_precision:.4f}")
    print(f"  Recall:      {primary_recall:.4f}")
    print(f"  F1 Score:    {primary_f1:.4f}")
    print(f"  Trades:      {primary_n_trades}")
    print()
    print("PRIMARY + META-MODEL:")
    print(f"  ROC-AUC:     {meta_auc:.4f}  ({meta_auc - primary_auc:+.4f})")
    print(f"  Precision:   {meta_precision:.4f}  ({precision_gain:+.1%})")
    print(f"  Recall:      {meta_recall:.4f}  ({recall_change:+.4f})")
    print(f"  F1 Score:    {meta_f1:.4f}  ({f1_change:+.4f})")
    print(f"  Trades:      {meta_n_trades}  ({trades_reduction:.1%} reduction)")
    print()
    print("IMPACT SUMMARY:")
    print(f"  Precision Gain:    {precision_gain:+.1%}")
    print(f"  Trade Reduction:   {trades_reduction:.1%}")
    print(f"  F1 Improvement:    {f1_change:+.4f}")
    print()
    print("="*80)

    # Save results
    results = {
        "interval": interval,
        "test_period": {"start": start, "end": end},
        "meta_threshold": meta_threshold,
        "n_samples": len(y_true_clean),
        "primary_only": {
            "roc_auc": primary_auc,
            "precision": primary_precision,
            "recall": primary_recall,
            "f1": primary_f1,
            "n_trades": int(primary_n_trades),
        },
        "primary_plus_meta": {
            "roc_auc": meta_auc,
            "precision": meta_precision,
            "recall": meta_recall,
            "f1": meta_f1,
            "n_trades": int(meta_n_trades),
        },
        "improvements": {
            "precision_gain": precision_gain,
            "recall_change": recall_change,
            "f1_change": f1_change,
            "trades_reduction": trades_reduction,
            "auc_change": meta_auc - primary_auc,
        }
    }

    results_path = Path("reports") / f"meta_evaluation_{interval}_{start}_{end}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
