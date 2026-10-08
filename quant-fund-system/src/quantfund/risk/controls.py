"""
Risk control utilities: volatility targeting, drawdowns, and loss stops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Interval = Literal["1d", "60m", "120m"]


def annualization_factor(interval: Interval) -> float:
    if interval == "1d":
        return 252.0
    if interval == "60m":
        return 252.0 * 6.5
    if interval == "120m":
        return 252.0 * 3.25
    return 252.0


def target_bar_sigma(annual_vol_target: float, interval: Interval) -> float:
    return float(annual_vol_target) / np.sqrt(annualization_factor(interval))


def per_trade_weight_cap(asset_bar_vol: float, per_trade_risk: float, min_vol: float = 1e-4) -> float:
    # Handle negative or invalid inputs
    if asset_bar_vol <= 0 or per_trade_risk <= 0:
        return 0.0
    vol = max(float(asset_bar_vol), min_vol)
    cap = float(per_trade_risk) / vol
    return float(np.clip(cap, 0.0, 1.0))


def portfolio_vol_proxy(weights: np.ndarray, asset_bar_vols: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(asset_bar_vols, dtype=float)
    return float(np.sqrt(np.sum((w * v) ** 2)))


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1.0
    return dd


def daily_loss_stopped(portfolio_returns: pd.Series, threshold: float) -> pd.Series:
    """Return a boolean series indicating if trading should stop for the rest of day.

    When cumulative return within a day falls below -threshold, mark subsequent bars that day True.
    """
    if portfolio_returns.empty:
        return pd.Series(dtype=bool, index=portfolio_returns.index)
    idx_utc = pd.to_datetime(portfolio_returns.index, utc=True)
    df = pd.DataFrame({"ret": portfolio_returns}, index=portfolio_returns.index)
    df["date"] = pd.DatetimeIndex([
        ts.replace(hour=0, minute=0, second=0, microsecond=0) for ts in idx_utc
    ])
    out = pd.Series(False, index=portfolio_returns.index)
    for d, grp in df.groupby("date"):
        cum = grp["ret"].cumsum()
        hit = cum <= (-float(threshold))
        if hit.any():
            first_idx = hit.idxmax()  # first True index
            # After first_idx within the day, stop
            mask = (df.index >= first_idx) & (df["date"] == d)
            out.loc[mask] = True
    return out


__all__ = [
    "annualization_factor",
    "target_bar_sigma",
    "per_trade_weight_cap",
    "portfolio_vol_proxy",
    "compute_drawdown",
    "daily_loss_stopped",
]