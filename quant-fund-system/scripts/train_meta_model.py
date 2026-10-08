#!/usr/bin/env python3
"""
Train meta-labeling model for bet sizing.

Two-stage approach:
1. Primary model (existing): Predicts direction
2. Meta-model (new): Predicts whether to bet on primary's prediction

Usage:
    python scripts/train_meta_model.py --start 2020-01-02 --end 2024-01-30 --interval 1d
"""
import click
import json
from pathlib import Path
from datetime import date
from typing import Dict

from quantfund.models.train import (
    load_train_config, _collect_parquets, _concat_datasets,
    _time_series_folds, _metrics_bin, write_calibration_plot,
    REQUIRED_COLS, BASE_DROP
)
from quantfund.models.meta_labeling import (
    generate_meta_labels,
    build_meta_features,
    apply_meta_model,
    evaluate_meta_labeling,
    MetaLabelConfig,
)
from quantfund.utils.log import get_logger

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from joblib import dump, load

logger = get_logger(__name__)


def load_primary_model(interval: str = "1d"):
    """Load trained primary model."""
    model_path = Path(f"artifacts/{interval}/model_all.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Primary model not found: {model_path}")

    model_data = load(model_path)

    # Handle both dict format (from triple_barrier) and raw model (from regular training)
    if isinstance(model_data, dict):
        return model_data['model'], model_data.get('calibrator')
    else:
        # Raw model object, no calibrator
        return model_data, None


def load_design_matrix_meta(
    start: date,
    end: date,
    interval: str = "1d",
):
    """Load data and generate primary predictions + meta-labels."""
    parts = _collect_parquets(interval)
    if not parts:
        return None, None, None, {" reason": "no parquet files", "interval": interval}

    df = _concat_datasets(parts)
    if df.empty:
        return None, None, None, {"reason": "empty concat", "interval": interval}

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            df = df.copy()
            df["timestamp"] = idx
        elif "datetime" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)

    # Check required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, None, None, {"reason": f"missing columns: {missing}", "interval": interval}

    # Filter by date
    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = (pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")) + pd.Timedelta(days=1)
    m = (ts >= start_ts) & (ts <= end_ts)
    df = df.loc[m].copy()
    if df.empty:
        return None, None, None, {"reason": "no rows after date filter", "interval": interval}

    # Load primary model
    logger.info("Loading primary model...")
    primary_model, calibrator = load_primary_model(interval)

    # Get primary model predictions
    logger.info("Generating primary model predictions...")
    drop_cols = set(BASE_DROP)
    numeric_cols = [c for c in df.columns if c not in drop_cols and c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])]

    X_primary = df[numeric_cols].astype(np.float32)
    primary_probs = primary_model.predict_proba(X_primary.values)[:, 1]

    if calibrator is not None:
        primary_probs = calibrator.transform(primary_probs)

    primary_probs = pd.Series(primary_probs, index=df.index)
    primary_side = (primary_probs >= 0.5).astype(int)

    # Generate meta-labels
    logger.info("Generating meta-labels using triple barrier...")
    config = MetaLabelConfig(
        profit_pct=0.09,
        stop_pct=0.09,
        max_holding_bars=29,
        profit_threshold=0.03,
        stop_threshold=-0.03,
        min_primary_prob=0.52,
    )

    meta_labels, barrier_results = generate_meta_labels(
        df, primary_probs, primary_side, config
    )

    logger.info(f"Meta-label distribution: {meta_labels.value_counts().to_dict()}")

    # Build meta-features
    logger.info("Building meta-features...")
    meta_features = build_meta_features(df, primary_probs)

    # Only keep rows with valid meta-labels
    ok = meta_labels.notna()
    X_meta = meta_features.loc[ok].copy()
    y_meta = meta_labels.loc[ok].astype(np.int8)
    ts_meta = df.loc[ok, "timestamp"].copy()

    logger.info(f"Meta-training samples: {len(X_meta)} (filtered from {len(df)})")

    meta = {
        "rows": int(len(X_meta)),
        "features": list(X_meta.columns),
        "interval": interval,
        "config": {
            "profit_pct": config.profit_pct,
            "stop_pct": config.stop_pct,
            "max_holding_bars": config.max_holding_bars,
            "profit_threshold": config.profit_threshold,
            "stop_threshold": config.stop_threshold,
            "min_primary_prob": config.min_primary_prob,
        },
    }

    # Attach timestamps for CV
    X_meta = X_meta.copy()
    X_meta["_timestamp"] = ts_meta.values

    return X_meta, y_meta, primary_probs, meta


@click.command()
@click.option("--start", required=True, type=str, help="Training start date (YYYY-MM-DD)")
@click.option("--end", required=True, type=str, help="Training end date (YYYY-MM-DD)")
@click.option("--interval", default="1d", type=click.Choice(["1d", "5d", "60m", "120m"]), show_default=True)
def main(start, end, interval):
    """Train meta-labeling model."""
    logger.info(f"Training meta-model: interval={interval}, period={start} to {end}")

    # Load data and generate meta-labels
    X, y, primary_probs, meta = load_design_matrix_meta(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        interval=interval,
    )

    if X is None or y is None:
        logger.error(f"Failed to load data: {meta.get('reason', 'unknown')}")
        print(json.dumps({"error": meta.get("reason"), "meta": meta}, indent=2))
        return

    logger.info(f"Loaded {len(X)} meta-samples with {len(meta['features'])} features")
    logger.info(f"Meta-label distribution: {pd.Series(y).value_counts().to_dict()}")

    # Extract timestamps for CV
    ts = X["_timestamp"]
    X_feat = X.drop(columns=["_timestamp"])
    feature_names = X_feat.columns.tolist()

    # Time-series CV
    train_config = load_train_config()
    folds = _time_series_folds(ts, train_config.cv_splits, train_config.purge_days, train_config.embargo_days)

    logger.info(f"Created {len(folds)} CV folds (purge={train_config.purge_days}, embargo={train_config.embargo_days})")

    # Train models
    oof_proba = np.zeros(len(X_feat))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_train, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Check if both classes present in training
        if len(np.unique(y_train)) < 2:
            logger.warning(f"Fold {fold_idx}: Only one class in training set, skipping")
            continue

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train.values, y_train.values, verbose=False)

        # Predict probabilities
        val_proba = model.predict_proba(X_val.values)[:, 1]
        oof_proba[val_idx] = val_proba

        # Metrics
        metrics = _metrics_bin(y_val.values, val_proba)
        metrics.update({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        fold_metrics.append(metrics)

        logger.info(f"Fold {fold_idx}: ROC-AUC={metrics['roc_auc']:.4f}, n_train={len(train_idx)}, n_val={len(val_idx)}")

    # OOF metrics
    oof_mask = (oof_proba > 0)
    oof_metrics = _metrics_bin(y.values[oof_mask], oof_proba[oof_mask])
    oof_metrics["oof_n"] = int(oof_mask.sum())

    logger.info(f"OOF ROC-AUC (meta-model): {oof_metrics['roc_auc']:.4f}")

    # Calibrate probabilities
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_proba[oof_mask], y.values[oof_mask])
    oof_proba_cal = calibrator.transform(oof_proba[oof_mask])
    oof_metrics_cal = _metrics_bin(y.values[oof_mask], oof_proba_cal)

    logger.info(f"OOF ROC-AUC (calibrated): {oof_metrics_cal['roc_auc']:.4f}")

    # Train final model on all data
    logger.info("Training final meta-model on all data...")
    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_feat.values, y.values, verbose=False)

    # Save artifacts
    out_dir = Path(f"artifacts/{interval}_meta")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "meta_model.joblib"
    dump({'model': final_model, 'calibrator': calibrator}, model_path)
    logger.info(f"Saved meta-model to {model_path}")

    # Save feature metadata
    meta_out = {
        "features": feature_names,
        "config": meta["config"],
        "training_period": {"start": start, "end": end},
        "interval": interval,
    }
    meta_path = out_dir / "meta_feature_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    # Save calibration plot
    write_calibration_plot(
        y.values[oof_mask],
        oof_proba[oof_mask],
        out_dir="reports",
        filename=f"calibration_curve_{interval}_meta.png"
    )

    # Compile results
    results = {
        "interval": interval,
        "label_method": "meta_labeling",
        "config": meta["config"],
        "rows": meta["rows"],
        "features": len(feature_names),
        "cv": {
            "folds": fold_metrics,
            "oof": oof_metrics,
        },
        "metrics": {
            "oof_raw": oof_metrics,
            "oof_calibrated": oof_metrics_cal,
        },
        "artifacts": {
            "meta_model": str(model_path),
            "meta_feature_meta": str(meta_path),
        },
        "date_range": {"start": start, "end": end},
    }

    # Save summary
    summary_path = Path("reports") / f"train_summary_{interval}_meta.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved training summary to {summary_path}")

    # Print results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
