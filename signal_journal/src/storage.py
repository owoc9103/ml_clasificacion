"""Historial acumulado en CSV / Excel."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CSV_PATH, DATA_DIR, EXCEL_PATH


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> pd.DataFrame:
    ensure_data_dir()
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH, parse_dates=["run_ts", "feature_date"], low_memory=False)


def has_run_for_date(session_date: date) -> bool:
    hist = load_history()
    if hist.empty or "session_date" not in hist.columns:
        return False
    return session_date.isoformat() in set(hist["session_date"].astype(str))


def append_snapshot(df: pd.DataFrame, *, replace_session: bool = False) -> Path:
    """Añade filas al CSV (una fila por símbolo por consulta)."""
    ensure_data_dir()
    if df.empty:
        raise ValueError("No hay filas para guardar.")

    out = df.copy()
    if replace_session and CSV_PATH.exists():
        hist = load_history()
        if not hist.empty and "session_date" in hist.columns:
            drop_day = out["session_date"].iloc[0]
            hist = hist[hist["session_date"].astype(str) != str(drop_day)]
            combined = pd.concat([hist, out], ignore_index=True)
        else:
            combined = out
    else:
        if CSV_PATH.exists():
            combined = pd.concat([load_history(), out], ignore_index=True)
        else:
            combined = out

    combined.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    try:
        combined.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    except Exception:
        pass
    return CSV_PATH


def latest_session() -> Optional[pd.DataFrame]:
    hist = load_history()
    if hist.empty:
        return None
    last = hist["session_date"].astype(str).max()
    return hist[hist["session_date"].astype(str) == last].copy()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
