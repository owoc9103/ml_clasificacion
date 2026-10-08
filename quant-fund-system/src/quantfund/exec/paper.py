"""
Paper executor:
- If ALPACA_API_KEY and ALPACA_SECRET exist, prepare for Alpaca paper trading
- Otherwise, run local simulation and emit signals at 15:45 ET based on current features

Note: This is a minimal scaffold; order routing not implemented here.
"""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from quantfund.features.core import build_features_for_symbol


def _has_alpaca_keys() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET"))


def generate_signals_1545_et(features_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Select last bar before 15:45 ET; assume index in UTC or tz-aware
    rows = []
    for sym, df in features_by_symbol.items():
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            continue
        idx = df.index
        if idx.tz is None:
            df_et = df.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            df_et = df.tz_convert("US/Eastern")
        # Build cutoff times per row day at 15:45 ET using replace
        cutoff_index = pd.DatetimeIndex(
            [ts.replace(hour=15, minute=45, second=0, microsecond=0) for ts in df_et.index]
        )
        mask = df_et.index <= cutoff_index
        last = df.loc[mask].tail(1)
        if last.empty:
            continue
        ema20_z = float(last.get("ema20_z", pd.Series([0.0], index=last.index)).iloc[0])
        p_up = 1.0 / (1.0 + pow(2.718281828, -ema20_z))
        weight = max(0.0, min(1.0, (p_up - 0.5) * 2.0))
        rows.append({"symbol": sym, "weight": weight})
    return pd.DataFrame(rows)


def main() -> None:
    if _has_alpaca_keys():
        print("Alpaca keys detected. Integrate with Alpaca paper trading here.")
        return
    print("No Alpaca keys found. Running local simulation placeholder.")


if __name__ == "__main__":
    main()