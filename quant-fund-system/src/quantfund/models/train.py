# src/quantfund/models/train.py
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from joblib import dump
from datetime import date

from quantfund.models.labels import create_labels, LabelType


# ---------- Config & IO ----------

@dataclass(frozen=True)
class TrainConfig:
    cost_bps: float = 2.0
    cv_splits: int = 5
    purge_days: int = 5
    embargo_days: int = 2


def _read_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_config() -> TrainConfig:
    cfg = _read_yaml_config(Path("configs/train.yaml"))
    return TrainConfig(
        cost_bps=float(cfg.get("cost_bps", 2.0)),
        cv_splits=int(cfg.get("cv_splits", 5)),
        purge_days=int(cfg.get("purge_days", 5)),
        embargo_days=int(cfg.get("embargo_days", 2)),
    )


def get_cost_from_config() -> float:
    """Return cost as decimal (e.g., 2 bps -> 0.0002)."""
    tc = load_train_config()
    return tc.cost_bps / 1e4


# ---------- Dataset loading & labeling ----------

REQUIRED_COLS = {"timestamp", "symbol", "open", "close", "next_open", "next_close"}
BASE_DROP = {"ret_next", "next_open", "next_close", "open", "high", "low", "close", "volume", "symbol"}

def _collect_parquets(interval: str) -> List[Path]:
    base = Path(f"data/datasets/interval={interval}")
    if not base.exists():
        return []
    return sorted(base.rglob("*.parquet"))

def _concat_datasets(parts: List[Path]) -> pd.DataFrame:
    frames = []
    for p in parts:
        try:
            df = pd.read_parquet(p)
            # Preserve time information by materializing index as 'timestamp'
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
                df = df.copy()
                df["timestamp"] = idx
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

def load_processed_design_matrix(
    start: date,
    end: date,
    interval: str = "1d",
    label_type: LabelType = "binary",
    lookforward: int = 10,
) -> Tuple[pd.DataFrame | None, pd.Series | None, Dict]:
    parts = _collect_parquets(interval)
    if not parts:
        return None, None, {"reason": "no parquet files", "interval": interval}

    df = _concat_datasets(parts)
    if df.empty:
        return None, None, {"reason": "empty concat", "interval": interval}

    # Ensure timestamp column exists (parquets may store it as DatetimeIndex or 'datetime' column)
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            # Preserve index and add explicit timestamp column in UTC
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

    # Basic checks
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, None, {"reason": f"missing columns: {missing}", "interval": interval}

    # Filter by date (ensure tz-aware on both sides)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = (pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")) + pd.Timedelta(days=1)
    m = (ts >= start_ts) & (ts <= end_ts)
    df = df.loc[m].copy()
    if df.empty:
        return None, None, {"reason": "no rows after date filter", "interval": interval}

    # Label using specified strategy
    cost = get_cost_from_config()
    y = create_labels(df, label_type=label_type, cost=cost, lookforward=lookforward)

    # Drop rows without valid label
    ok = y.notna()
    if "next_open" in df.columns and "next_close" in df.columns:
        ok = ok & df["next_open"].notna() & df["next_close"].notna()
    df = df.loc[ok].copy()
    y = y.loc[ok].copy()

    # Feature selection: keep numeric features excluding base/raw columns
    drop_cols = set(BASE_DROP)
    # Keep timestamp for CV; will drop from X
    numeric_cols = [c for c in df.columns if c not in drop_cols and c not in {"timestamp"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None, None, {"reason": "no numeric features", "interval": interval}

    X = df[numeric_cols].astype(np.float32)
    y = y.astype(np.int8)
    meta = {
        "rows": int(len(X)),
        "features": numeric_cols,
        "interval": interval,
        "symbols": sorted(map(str, set(df.get("symbol", pd.Series(dtype=str)).astype(str).tolist()))),
        "cost_bps": load_train_config().cost_bps,
    }
    # Attach timestamps for CV splitting
    X = X.copy()
    X["_timestamp"] = df["timestamp"].values
    return X, y, meta


# ---------- Time-series CV with purge & embargo ----------

def _time_series_folds(ts: pd.Series, n_splits: int, purge_days: int, embargo_days: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build time-ordered folds using unique days; apply purge & embargo around each validation fold.
    Returns list of (train_idx, val_idx).
    """
    # Convert timestamps to date (day resolution)
    days = pd.to_datetime(ts).dt.floor("D")
    uniq_days = days.sort_values().unique()
    if len(uniq_days) < n_splits + 1:
        n_splits = max(2, min(3, len(uniq_days) - 1))

    folds = []
    # simple expanding window over unique days
    chunk = len(uniq_days) // (n_splits + 1)
    if chunk == 0:
        chunk = 1
    for k in range(1, n_splits + 1):
        train_end_day = uniq_days[k * chunk] if (k * chunk) < len(uniq_days) else uniq_days[-2]
        val_start_idx = k * chunk
        val_end_idx = min(len(uniq_days) - 1, val_start_idx + chunk)

        val_days = uniq_days[val_start_idx:val_end_idx]
        if len(val_days) == 0:
            continue

        # Purge & embargo
        purge_delta = pd.Timedelta(days=purge_days)
        embargo_delta = pd.Timedelta(days=embargo_days)

        val_start = pd.Timestamp(val_days[0])
        val_end = pd.Timestamp(val_days[-1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        purge_start = val_start - purge_delta
        purge_end = val_end + embargo_delta

        train_mask = (days < purge_start) | (days > purge_end)
        val_mask = (days >= val_start) & (days <= val_end)

        tr_idx = np.where(train_mask.values)[0]
        va_idx = np.where(val_mask.values)[0]

        # Ensure non-empty
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append((tr_idx, va_idx))

    return folds


# ---------- Metrics & plots ----------

def _metrics_bin(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=[0,1]))
    except ValueError:
        out["log_loss"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, proba))
    except ValueError:
        out["brier"] = float("nan")
    return out

def write_calibration_plot(y_true: np.ndarray, proba: np.ndarray, out_dir: str = "reports", filename: str = "calibration_curve.png") -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Decile reliability plot
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(proba, bins) - 1
    avg_pred = []
    avg_true = []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            avg_pred.append(proba[m].mean())
            avg_true.append(y_true[m].mean())
    plt.figure(figsize=(5,5))
    plt.plot([0,1], [0,1], linestyle="--")
    if avg_pred:
        plt.scatter(avg_pred, avg_true)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    fp = str(out_path / filename)
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    return fp


# ---------- Train & Calibrate ----------

def timeseries_cv_with_purge_embargo(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> Tuple[Dict, np.ndarray]:
    ts = pd.to_datetime(X["_timestamp"])
    folds = _time_series_folds(ts, cfg.cv_splits, cfg.purge_days, cfg.embargo_days)
    if not folds:
        return {"error": "no folds"}, np.array([])

    oof_proba = np.full(len(X), np.nan, dtype=np.float32)
    reports = []

    # Define base model
    base_params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,
        tree_method="hist",
        random_state=42
    )

    feat_cols = [c for c in X.columns if c != "_timestamp"]

    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        Xtr = X.iloc[tr_idx][feat_cols].values
        ytr = y.iloc[tr_idx].values
        Xva = X.iloc[va_idx][feat_cols].values
        yva = y.iloc[va_idx].values

        model = XGBClassifier(**base_params)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        oof_proba[va_idx] = p

        rep = _metrics_bin(yva, p)
        rep["fold"] = i
        rep["n_train"] = int(len(tr_idx))
        rep["n_val"] = int(len(va_idx))
        reports.append(rep)

    # Aggregate OOF metrics
    valid = ~np.isnan(oof_proba)
    if valid.sum() == 0:
        return {"error": "no oof predictions"}, np.array([])

    agg = _metrics_bin(y.iloc[valid].values, oof_proba[valid])
    agg["oof_n"] = int(valid.sum())
    return {"folds": reports, "oof": agg}, oof_proba[valid]


def fit_isotonic_on_oof(oof_proba: np.ndarray, y_true: pd.Series) -> IsotonicRegression:
    # y_true must be aligned with oof_proba valid mask; handled in caller
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(oof_proba, y_true.values)
    return ir


def compute_metrics(y_true: pd.Series, proba: np.ndarray) -> Dict[str, float]:
    y_np = np.asarray(y_true.to_numpy())
    return _metrics_bin(y_np, proba)


# ---------- Walk-forward retrain & save artifacts ----------

def _quarter_label(ts: pd.Timestamp) -> str:
    q = (ts.quarter)
    return f"{ts.year}Q{q}"

def walk_forward_train_and_save(X: pd.DataFrame, y: pd.Series, meta: Dict, wf_quarters: int = 4, out_dir: str = "artifacts") -> Dict:
    """
    Retrain a model on each of the last N quarters and save JSON meta + calibrator.
    (Basit versiyon: tek bir toplu model de ekliyoruz.)
    """
    out = {}
    out_path = Path(out_dir) / meta["interval"]
    out_path.mkdir(parents=True, exist_ok=True)

    # Toplu model (tüm veri)
    feat_cols = [c for c in X.columns if c != "_timestamp"]
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", n_jobs=0, tree_method="hist", random_state=42
    )
    model.fit(X[feat_cols].values, y.values)
    dump(model, out_path / "model_all.joblib")

    # Quarter-based retrain (son N çeyrek)
    ts = pd.to_datetime(X["_timestamp"], errors="coerce")
    quarters = pd.PeriodIndex(ts, freq="Q").astype(str)
    uq = pd.unique(quarters)
    if len(uq) > 0:
        last_q = list(uq)[-wf_quarters:]
        saved = []
        for q in last_q:
            m = (quarters == q)
            if m.sum() < 200:
                continue
            mq = X.loc[m, feat_cols].values
            yq = y.loc[m].values
            m_q = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.07,
                subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
                eval_metric="logloss", n_jobs=0, tree_method="hist", random_state=42
            )
            m_q.fit(mq, yq)
            fp = out_path / f"model_{q}.joblib"
            dump(m_q, fp)
            saved.append(str(fp))
        out["quarter_models"] = saved

    # Meta yaz
    meta_fp = out_path / "feature_meta.json"
    with open(meta_fp, "w", encoding="utf-8") as f:
        json.dump({"features": [c for c in X.columns if c != "_timestamp"]}, f, indent=2)

    out["model_all"] = str(out_path / "model_all.joblib")
    out["feature_meta"] = str(meta_fp)
    return out


# ---------- Public entrypoint ----------

def train_main(
    start: date,
    end: date,
    wf_quarters: int = 4,
    interval: str = "1d",
    label_type: LabelType = "binary",
    lookforward: int = 10,
    model_name: str = "xgboost",
) -> dict:
    from quantfund.models.multi_train import train_one_model

    _ = wf_quarters  # reservado; walk-forward por trimestre solo en flujo legacy XGBoost
    out = train_one_model(
        start,
        end,
        model_name=model_name,
        interval=interval,
        label_type=label_type,
        lookforward=lookforward,
    )
    if "error" in out:
        return out
    Path("reports").mkdir(parents=True, exist_ok=True)
    summary = {
        "interval": out["interval"],
        "rows": out["rows"],
        "features": out["features"],
        "cv": {"folds": out.get("cv_folds", []), "oof": out["metrics"]["oof_raw"]},
        "metrics": out["metrics"],
        "artifacts": out["artifacts"],
        "calibration_curve": out["calibration_curve"],
        "date_range": out["date_range"],
        "model": model_name,
    }
    with open(Path("reports") / f"train_summary_{interval}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
