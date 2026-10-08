#!/usr/bin/env python3
"""
Train model with triple barrier labels instead of binary labels.

Usage:
    python scripts/train_triple_barrier.py --start 2020-01-02 --end 2024-01-30 --interval 1d
"""
import click
import json
from pathlib import Path
from datetime import date
from typing import Dict

from quantfund.models.train import (
    load_train_config, _collect_parquets, _concat_datasets,
    _time_series_folds, _metrics_bin, write_calibration_plot,
    get_cost_from_config, REQUIRED_COLS, BASE_DROP
)
from quantfund.models.triple_barrier import triple_barrier_labels, TripleBarrierConfig
from quantfund.utils.log import get_logger

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from joblib import dump

logger = get_logger(__name__)


def load_design_matrix_triple_barrier(
    start: date,
    end: date,
    interval: str = "1d",
    tb_config: TripleBarrierConfig = None,
):
    """
    Load data and create labels using triple barrier method.

    Similar to load_processed_design_matrix but uses triple barrier labels.
    """
    parts = _collect_parquets(interval)
    if not parts:
        return None, None, {"reason": "no parquet files", "interval": interval}

    df = _concat_datasets(parts)
    if df.empty:
        return None, None, {"reason": "empty concat", "interval": interval}

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
        return None, None, {"reason": f"missing columns: {missing}", "interval": interval}

    # Filter by date
    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = (pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")) + pd.Timedelta(days=1)
    m = (ts >= start_ts) & (ts <= end_ts)
    df = df.loc[m].copy()
    if df.empty:
        return None, None, {"reason": "no rows after date filter", "interval": interval}

    # Triple barrier labels
    if tb_config is None:
        tb_config = TripleBarrierConfig()

    # Group by symbol and apply triple barrier per symbol
    logger.info(f"Applying triple barrier labeling (profit={tb_config.profit_pct:.1%}, stop={tb_config.stop_pct:.1%}, holding={tb_config.max_holding_bars} bars)")

    all_labels = []
    for symbol, group in df.groupby("symbol"):
        group_sorted = group.sort_values("timestamp")
        labels = triple_barrier_labels(group_sorted, tb_config)
        all_labels.append(labels)

    y = pd.concat(all_labels)

    # Convert labels from {-1, 0, 1} to {0, 1, 2} for XGBoost multiclass
    # -1 (short) → 0
    #  0 (neutral) → 1
    #  1 (long) → 2
    y_mapped = y.copy()
    y_mapped[y == -1] = 0
    y_mapped[y == 0] = 1
    y_mapped[y == 1] = 2

    # Drop rows without valid label
    ok = y_mapped.notna()
    df = df.loc[ok].copy()
    y_mapped = y_mapped.loc[ok].copy()

    # Feature selection
    drop_cols = set(BASE_DROP)
    numeric_cols = [c for c in df.columns if c not in drop_cols and c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None, None, {"reason": "no numeric features", "interval": interval}

    X = df[numeric_cols].astype(np.float32)
    y_final = y_mapped.astype(np.int8)

    meta = {
        "rows": int(len(X)),
        "features": numeric_cols,
        "interval": interval,
        "symbols": sorted(map(str, set(df.get("symbol", pd.Series(dtype=str)).astype(str).tolist()))),
        "label_method": "triple_barrier",
        "tb_config": {
            "profit_pct": tb_config.profit_pct,
            "stop_pct": tb_config.stop_pct,
            "max_holding_bars": tb_config.max_holding_bars,
        },
    }

    # Attach timestamps for CV
    X = X.copy()
    X["_timestamp"] = df["timestamp"].values
    return X, y_final, meta


@click.command()
@click.option("--start", required=True, type=str, help="Training start date (YYYY-MM-DD)")
@click.option("--end", required=True, type=str, help="Training end date (YYYY-MM-DD)")
@click.option("--interval", default="1d", type=click.Choice(["1d", "5d", "60m", "120m"]), show_default=True)
@click.option("--profit-pct", default=0.09, type=float, show_default=True, help="Profit take percentage")
@click.option("--stop-pct", default=0.09, type=float, show_default=True, help="Stop loss percentage")
@click.option("--holding-bars", default=29, type=int, show_default=True, help="Max holding period in bars")
def main(start, end, interval, profit_pct, stop_pct, holding_bars):
    """Train model with triple barrier labels."""
    logger.info(f"Training with triple barrier: interval={interval}, period={start} to {end}")

    # Create triple barrier config
    tb_config = TripleBarrierConfig(
        profit_pct=profit_pct,
        stop_pct=stop_pct,
        max_holding_bars=holding_bars,
    )

    # Load data
    X, y, meta = load_design_matrix_triple_barrier(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        interval=interval,
        tb_config=tb_config,
    )

    if X is None or y is None:
        logger.error(f"Failed to load data: {meta.get('reason', 'unknown')}")
        print(json.dumps({"error": meta.get("reason"), "meta": meta}, indent=2))
        return

    logger.info(f"Loaded {len(X)} samples with {len(meta['features'])} features")
    logger.info(f"Label distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    # Extract timestamps for CV
    ts = X["_timestamp"]
    X_feat = X.drop(columns=["_timestamp"])
    feature_names = X_feat.columns.tolist()

    # Time-series CV
    train_config = load_train_config()
    folds = _time_series_folds(ts, train_config.cv_splits, train_config.purge_days, train_config.embargo_days)

    logger.info(f"Created {len(folds)} CV folds (purge={train_config.purge_days}, embargo={train_config.embargo_days})")

    # Train models
    oof_proba = np.zeros((len(X_feat), 3))  # 3 classes
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        X_train, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective='multi:softprob',  # Multiclass
            num_class=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train.values, y_train.values, verbose=False)

        # Predict probabilities
        val_proba = model.predict_proba(X_val.values)
        oof_proba[val_idx] = val_proba

        # Metrics (use class 2 probability for "long" signal)
        metrics = _metrics_bin(y_val.values == 2, val_proba[:, 2])
        metrics.update({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        fold_metrics.append(metrics)

        logger.info(f"Fold {fold_idx}: ROC-AUC={metrics['roc_auc']:.4f}, n_train={len(train_idx)}, n_val={len(val_idx)}")

    # OOF metrics (long class)
    oof_mask = (oof_proba.sum(axis=1) > 0)
    oof_metrics = _metrics_bin((y.values == 2)[oof_mask], oof_proba[oof_mask, 2])
    oof_metrics["oof_n"] = int(oof_mask.sum())

    logger.info(f"OOF ROC-AUC (long class): {oof_metrics['roc_auc']:.4f}")

    # Calibrate probabilities
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_proba[oof_mask, 2], (y.values == 2)[oof_mask])
    oof_proba_cal = calibrator.transform(oof_proba[oof_mask, 2])
    oof_metrics_cal = _metrics_bin((y.values == 2)[oof_mask], oof_proba_cal)

    logger.info(f"OOF ROC-AUC (calibrated): {oof_metrics_cal['roc_auc']:.4f}")

    # Train final model on all data
    logger.info("Training final model on all data...")
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_feat.values, y.values, verbose=False)

    # Save artifacts
    out_dir = Path(f"artifacts/{interval}_triple_barrier")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model_all.joblib"
    dump({'model': final_model, 'calibrator': calibrator}, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save feature metadata
    meta_out = {
        "features": feature_names,
        "label_method": "triple_barrier",
        "tb_config": {
            "profit_pct": profit_pct,
            "stop_pct": stop_pct,
            "max_holding_bars": holding_bars,
        },
        "training_period": {"start": start, "end": end},
        "interval": interval,
    }
    meta_path = out_dir / "feature_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    # Save calibration plot
    write_calibration_plot(
        (y.values == 2)[oof_mask],
        oof_proba[oof_mask, 2],
        out_dir="reports",
        filename=f"calibration_curve_{interval}_triple_barrier.png"
    )

    # Compile results
    results = {
        "interval": interval,
        "label_method": "triple_barrier",
        "tb_config": meta_out["tb_config"],
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
            "model_all": str(model_path),
            "feature_meta": str(meta_path),
        },
        "date_range": {"start": start, "end": end},
    }

    # Save summary
    summary_path = Path("reports") / f"train_summary_{interval}_triple_barrier.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved training summary to {summary_path}")

    # Print results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
