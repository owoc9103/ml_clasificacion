"""Calcula señales usando el quant-fund-system instalado o en src/."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import load

from src.config import (
    DISPLAY_INDICATORS,
    LIVE_CONFIG,
    QUANT_FUND_ROOT,
    TRAIN_CONFIG,
    UNIVERSE_CONFIG,
)


def _bootstrap_quantfund() -> None:
    src = QUANT_FUND_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(QUANT_FUND_ROOT)


def load_symbols(*, exclude_vix: bool = True) -> list[str]:
    with open(UNIVERSE_CONFIG, encoding="utf-8") as handle:
        uni = yaml.safe_load(handle) or {}
    symbols = list(uni.get("symbols") or [])
    if exclude_vix:
        symbols = [s for s in symbols if s != "^VIX"]
    return symbols


def _load_live_model(interval: str = "1d"):
    _bootstrap_quantfund()
    from quantfund.models.multi_train import resolve_live_model_path
    from quantfund.models.train import _read_yaml_config

    cfg = _read_yaml_config(TRAIN_CONFIG)
    model_key = str(cfg.get("live_model", "xgboost"))
    model_path, meta_path = resolve_live_model_path(interval, model_key)
    if not model_path.exists():
        raise FileNotFoundError(f"No hay modelo en {model_path}. Entrena primero en quant-fund-system.")
    model = load(model_path)
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    feature_cols = meta["features"]
    return model, feature_cols, model_key


def _load_momo_weight() -> float:
    with open(LIVE_CONFIG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return float(cfg.get("strategy", {}).get("momo_weight", 0.3))


def _load_limits() -> tuple[float, float, float]:
    with open(LIVE_CONFIG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    lim = cfg.get("limits") or {}
    return (
        float(lim.get("max_position_size", 0.30)),
        float(lim.get("min_position_size", 0.01)),
        float(cfg.get("strategy", {}).get("momo_weight", 0.3)),
    )


def _feature_row(symbol: str, interval: str = "1d") -> pd.Series | None:
    path = QUANT_FUND_ROOT / f"data/datasets/interval={interval}/symbol={symbol}/data.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    row = df.iloc[-1]
    if isinstance(df.index, pd.DatetimeIndex):
        row = row.copy()
        row["_feature_ts"] = df.index[-1]
    return row


def _signal_from_row(
    row: pd.Series,
    model,
    feature_cols: list[str],
    momo_weight: float,
    max_pos: float,
    min_pos: float,
) -> dict[str, Any]:
    from quantfund.models.registry import predict_positive_proba

    X = pd.DataFrame([row[feature_cols].fillna(0.0)])
    proba = float(predict_positive_proba(model, X)[0])

    momo_sig = (proba - 0.45) * (1.0 / 0.55) if proba > 0.45 else 0.0

    rsi2 = float(row.get("rsi_2", 50.0) or 50.0)
    rsi14 = float(row.get("rsi_14", 50.0) or 50.0)
    ema20_z = float(row.get("ema20_z", 0.0) or 0.0)

    rsi2_oversold = (10 - min(rsi2, 10)) / 10.0
    rsi14_oversold = (30 - min(rsi14, 30)) / 30.0
    oversold = rsi2_oversold * 0.5 + rsi14_oversold * 0.5
    meanrev_sig = oversold * 0.7 + max(0.0, -ema20_z / 3.0) * 0.3

    signal = float(np.clip(momo_sig * momo_weight + meanrev_sig * (1.0 - momo_weight), 0.0, 1.0))

    return {
        "ml_proba": proba,
        "momo_sig": float(momo_sig),
        "meanrev_sig": float(meanrev_sig),
        "signal": signal,
        "_max_pos": max_pos,
        "_min_pos": min_pos,
    }


def _signals_to_weights(rows: list[dict[str, Any]]) -> None:
    total = sum(r["signal"] for r in rows)
    if total <= 0:
        for r in rows:
            r["target_weight"] = 0.0
            r["rank"] = 999
        return

    max_pos = rows[0]["_max_pos"]
    min_pos = rows[0]["_min_pos"]

    for r in rows:
        raw = r["signal"] / total
        w = float(np.clip(raw, 0.0, max_pos))
        if w < min_pos:
            w = 0.0
        r["raw_weight"] = raw
        r["target_weight"] = w

    total_w = sum(r["target_weight"] for r in rows)
    if total_w > 0:
        for r in rows:
            r["target_weight"] = r["target_weight"] / total_w * 0.95

    ranked = sorted(rows, key=lambda x: x["target_weight"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i


def build_daily_report(
    symbols: list[str] | None = None,
    *,
    session_date: date | None = None,
    notional_account: float = 100_000.0,
) -> pd.DataFrame:
    """Tabla por símbolo: indicadores, señales y peso sugerido (sin órdenes)."""
    _bootstrap_quantfund()
    symbols = symbols or load_symbols()
    model, feature_cols, model_key = _load_live_model()
    momo_weight = _load_momo_weight()
    max_pos, min_pos, _ = _load_limits()

    session_date = session_date or datetime.now(timezone.utc).date()
    run_ts = datetime.now(timezone.utc).replace(microsecond=0)

    records: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []

    for symbol in symbols:
        row = _feature_row(symbol)
        base: dict[str, Any] = {
            "session_date": session_date.isoformat(),
            "run_ts": run_ts.isoformat(),
            "symbol": symbol,
            "model": model_key,
            "suggested_notional_usd": 0.0,
            "data_ok": False,
        }
        if row is None:
            base["note"] = "sin dataset; ejecuta fetch_data + make_dataset"
            records.append(base)
            continue

        sig = _signal_from_row(row, model, feature_cols, momo_weight, max_pos, min_pos)
        feat_ts = row.get("_feature_ts", row.name if hasattr(row, "name") else None)
        base["data_ok"] = True
        base["feature_date"] = pd.Timestamp(feat_ts).isoformat() if feat_ts is not None else ""
        base.update({k: sig[k] for k in ("ml_proba", "momo_sig", "meanrev_sig", "signal")})
        for col in DISPLAY_INDICATORS:
            if col in row.index:
                base[f"ind_{col}"] = float(row[col]) if pd.notna(row[col]) else None
        partial.append({**base, **sig})
        records.append(base)

    _signals_to_weights(partial)
    weight_map = {r["symbol"]: r for r in partial}
    for rec in records:
        sym = rec["symbol"]
        if sym in weight_map:
            rec["target_weight"] = weight_map[sym].get("target_weight", 0.0)
            rec["rank"] = weight_map[sym].get("rank", 999)
            rec["suggested_notional_usd"] = round(
                float(rec.get("target_weight") or 0) * notional_account, 2
            )
        else:
            rec["target_weight"] = 0.0
            rec["rank"] = 999

    df = pd.DataFrame(records)
    if "rank" in df.columns:
        df = df.sort_values(["rank", "target_weight"], ascending=[True, False])
    return df
