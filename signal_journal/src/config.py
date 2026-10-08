"""Rutas al quant-fund-system (sin modificar ese repo)."""

from __future__ import annotations

from pathlib import Path

SIGNAL_JOURNAL_ROOT = Path(__file__).resolve().parents[1]
QUANT_FUND_ROOT = SIGNAL_JOURNAL_ROOT.parent / "quant-fund-system"
DATA_DIR = SIGNAL_JOURNAL_ROOT / "data"
CSV_PATH = DATA_DIR / "signal_journal.csv"
EXCEL_PATH = DATA_DIR / "signal_journal.xlsx"

LIVE_CONFIG = QUANT_FUND_ROOT / "configs" / "live_trading.yaml"
UNIVERSE_CONFIG = QUANT_FUND_ROOT / "configs" / "universe.yaml"
TRAIN_CONFIG = QUANT_FUND_ROOT / "configs" / "train.yaml"

# Indicadores clave para la vista matutina (el resto va en columnas extra si existen)
DISPLAY_INDICATORS = (
    "rsi_2",
    "rsi_14",
    "ema20_z",
    "r_5",
    "r_20",
    "macd_hist",
    "adx_14",
    "sharpe_20",
    "%b",
    "vol_ratio",
    "atr_14",
    "close",
)
