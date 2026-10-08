"""Entrenamiento y comparación multi-modelo con la misma validación temporal."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from quantfund.models.labels import LabelType
from quantfund.models.registry import (
    DISPLAY_NAMES,
    MODEL_NAMES,
    build_model,
    predict_positive_proba,
)
from quantfund.models.train import (
    TrainConfig,
    _metrics_bin,
    _time_series_folds,
    load_processed_design_matrix,
    load_train_config,
    write_calibration_plot,
)


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    p = np.clip(proba.astype(float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)

    def safe(fn):
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    base = _metrics_bin(y, p)
    base.update(
        {
            "accuracy": safe(lambda: accuracy_score(y, pred)),
            "balanced_accuracy": safe(lambda: balanced_accuracy_score(y, pred)),
            "precision": safe(lambda: precision_score(y, pred, zero_division=0)),
            "recall": safe(lambda: recall_score(y, pred, zero_division=0)),
            "f1": safe(lambda: f1_score(y, pred, zero_division=0)),
        }
    )
    return base


def _oof_predictions(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig, model_name: str
) -> tuple[np.ndarray, list[dict]]:
    ts = pd.to_datetime(X["_timestamp"])
    folds = _time_series_folds(ts, cfg.cv_splits, cfg.purge_days, cfg.embargo_days)
    feat_cols = [c for c in X.columns if c != "_timestamp"]
    oof = np.full(len(X), np.nan, dtype=np.float32)
    reports: list[dict] = []
    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        model = build_model(model_name)
        Xtr = X.iloc[tr_idx][feat_cols]
        ytr = y.iloc[tr_idx].astype(int)
        Xva = X.iloc[va_idx][feat_cols]
        yva = y.iloc[va_idx].astype(int)
        model.fit(Xtr, ytr)
        p = predict_positive_proba(model, Xva)
        oof[va_idx] = p
        rep = _metrics_bin(yva.values, p)
        rep.update({"fold": i, "n_train": len(tr_idx), "n_val": len(va_idx)})
        reports.append(rep)
    return oof, reports


def _artifact_dir(interval: str, model_name: str) -> Path:
    return Path("artifacts") / interval / "models" / model_name


def _save_artifacts(
    model_name: str,
    interval: str,
    model: Any,
    calibrator: IsotonicRegression | None,
    feature_cols: list[str],
    meta: dict,
) -> dict[str, str]:
    out_path = _artifact_dir(interval, model_name)
    out_path.mkdir(parents=True, exist_ok=True)
    model_fp = out_path / "model_all.joblib"
    dump(model, model_fp)
    paths = {"model_all": str(model_fp)}
    if calibrator is not None:
        cal_fp = out_path / "calibrator.joblib"
        dump(calibrator, cal_fp)
        paths["calibrator"] = str(cal_fp)
    meta_fp = out_path / "feature_meta.json"
    with open(meta_fp, "w", encoding="utf-8") as handle:
        json.dump({"features": feature_cols, "model": model_name}, handle, indent=2)
    paths["feature_meta"] = str(meta_fp)

    # Compatibilidad: XGBoost sigue en la ruta legacy que usa live_engine antiguo
    if model_name == "xgboost":
        legacy = Path("artifacts") / interval
        legacy.mkdir(parents=True, exist_ok=True)
        dump(model, legacy / "model_all.joblib")
        with open(legacy / "feature_meta.json", "w", encoding="utf-8") as handle:
            json.dump({"features": feature_cols, "model": model_name}, handle, indent=2)
        paths["legacy_model_all"] = str(legacy / "model_all.joblib")
    return paths


def train_one_model(
    start: date,
    end: date,
    model_name: str,
    interval: str = "1d",
    label_type: LabelType = "binary",
    lookforward: int = 10,
    wf_quarters: int = 0,
) -> dict:
    if model_name not in MODEL_NAMES:
        return {"error": f"unknown model {model_name}"}

    X, y, meta = load_processed_design_matrix(start, end, interval, label_type, lookforward)
    if X is None or y is None or len(X) == 0:
        return {"error": meta.get("reason", "design_matrix_empty"), "meta": meta}

    cfg = load_train_config()
    oof, fold_reports = _oof_predictions(X, y, cfg, model_name)
    valid = ~np.isnan(oof)
    if valid.sum() == 0:
        return {"error": "no oof predictions", "model": model_name}

    y_valid = y.iloc[valid]
    p_valid = oof[valid]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(p_valid, y_valid.values)
    p_cal = calibrator.predict(p_valid)

    feat_cols = [c for c in X.columns if c != "_timestamp"]
    final_model = build_model(model_name)
    final_model.fit(X[feat_cols], y.astype(int))
    art = _save_artifacts(model_name, interval, final_model, calibrator, feat_cols, meta)

    cal_plot = write_calibration_plot(
        y_valid.values,
        p_cal,
        out_dir="reports",
        filename=f"calibration_curve_{interval}_{model_name}.png",
    )

    metrics = {
        "oof_raw": _classification_metrics(y_valid.values, p_valid),
        "oof_calibrated": _classification_metrics(y_valid.values, p_cal),
    }

    result = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "interval": interval,
        "rows": meta["rows"],
        "features": len(feat_cols),
        "cv_folds": fold_reports,
        "metrics": metrics,
        "artifacts": art,
        "calibration_curve": cal_plot,
        "date_range": {"start": str(start), "end": str(end)},
    }
    return result


def train_all_models(
    start: date,
    end: date,
    interval: str = "1d",
    models: list[str] | None = None,
    label_type: LabelType = "binary",
    lookforward: int = 10,
) -> dict:
    names = list(models or MODEL_NAMES)
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for name in names:
        res = train_one_model(
            start, end, name, interval=interval, label_type=label_type, lookforward=lookforward
        )
        details[name] = res
        if "error" in res:
            continue
        m = res["metrics"]["oof_calibrated"]
        rows.append(
            {
                "model": DISPLAY_NAMES[name],
                "model_key": name,
                "roc_auc": m.get("roc_auc"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
                "log_loss": m.get("log_loss"),
                "brier": m.get("brier"),
            }
        )

    comparison = pd.DataFrame(rows)
    best = None
    if not comparison.empty and comparison["roc_auc"].notna().any():
        best = str(comparison.loc[comparison["roc_auc"].idxmax(), "model_key"])

    out = {
        "interval": interval,
        "comparison": comparison.to_dict(orient="records"),
        "best_predictive_model": best,
        "models": details,
        "date_range": {"start": str(start), "end": str(end)},
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    summary_fp = Path("reports") / f"model_comparison_{interval}.json"
    with open(summary_fp, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    out["comparison_file"] = str(summary_fp)
    return out


def resolve_live_model_path(interval: str, model_key: str | None = None) -> tuple[Path, Path]:
    """Ruta del modelo activo para paper trading."""
    cfg = load_train_config()
    key = model_key or _read_live_model_key()
    primary = _artifact_dir(interval, key) / "model_all.joblib"
    meta = _artifact_dir(interval, key) / "feature_meta.json"
    if primary.exists():
        return primary, meta
    legacy = Path("artifacts") / interval / "model_all.joblib"
    legacy_meta = Path("artifacts") / interval / "feature_meta.json"
    return legacy, legacy_meta


def _read_live_model_key() -> str:
    from quantfund.models.train import _read_yaml_config

    cfg = _read_yaml_config(Path("configs/train.yaml"))
    return str(cfg.get("live_model", "xgboost"))
