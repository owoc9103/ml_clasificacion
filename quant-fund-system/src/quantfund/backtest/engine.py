"""
Event-driven backtest engine with next-open fills, costs, and risk controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd

from quantfund.risk.controls import (
    target_bar_sigma,
    per_trade_weight_cap,
    portfolio_vol_proxy,
    compute_drawdown,
    daily_loss_stopped,
)

Interval = Literal["1d", "5d", "60m", "120m"]


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 2.0
    slippage_bps: float = 1.0

    def __post_init__(self):
        if self.commission_bps < 0:
            raise ValueError(f"commission_bps must be >= 0, got {self.commission_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if self.commission_bps > 100:
            raise ValueError(f"commission_bps must be <= 100, got {self.commission_bps}")
        if self.slippage_bps > 100:
            raise ValueError(f"slippage_bps must be <= 100, got {self.slippage_bps}")

    @property
    def roundtrip_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) * 1e-4


@dataclass(frozen=True)
class RiskLimits:
    annual_vol_target: float = 0.10
    max_drawdown: float = 0.12
    per_trade_risk: float = 0.005
    daily_loss_stop: float = 0.01
    atr_stop_mult: float = 3.0
    kelly_cap: float = 0.15  # 0.1–0.2 typical
    max_position_concentration: float = 0.30  # Max weight in single asset

    def __post_init__(self):
        if not 0 < self.annual_vol_target <= 1.0:
            raise ValueError(f"annual_vol_target must be in (0, 1], got {self.annual_vol_target}")
        if not 0 < self.max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {self.max_drawdown}")
        if not 0 < self.per_trade_risk <= 1.0:
            raise ValueError(f"per_trade_risk must be in (0, 1], got {self.per_trade_risk}")
        if not 0 < self.daily_loss_stop <= 1.0:
            raise ValueError(f"daily_loss_stop must be in (0, 1], got {self.daily_loss_stop}")
        if self.atr_stop_mult < 0:
            raise ValueError(f"atr_stop_mult must be >= 0, got {self.atr_stop_mult}")
        if not 0 <= self.kelly_cap <= 1.0:
            raise ValueError(f"kelly_cap must be in [0, 1], got {self.kelly_cap}")
        if not 0 < self.max_position_concentration <= 1.0:
            raise ValueError(f"max_position_concentration must be in (0, 1], got {self.max_position_concentration}")


@dataclass(frozen=True)
class BacktestConfig:
    interval: Interval
    cost: CostModel
    risk: RiskLimits


def _safe_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) == 0:
        return pd.Series(dtype=float, index=close.index)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def backtest_signals(
    prices_by_symbol: Dict[str, pd.DataFrame],
    signals_by_symbol: Dict[str, pd.Series],
    cfg: BacktestConfig,
) -> Dict[str, pd.DataFrame]:
    """Run event-driven backtest with next-open fills and risk controls.

    prices_by_symbol[symbol] columns: open, high, low, close
    signals_by_symbol[symbol]: desired raw signal in [0,1]

    Returns dict with:
      - timeseries: DataFrame with portfolio_ret, equity, drawdown
      - trades: DataFrame with executed trades
      - perf: DataFrame of summary metrics (single row)
    """
    # Common index
    idx = None
    for df in prices_by_symbol.values():
        if idx is None or len(df.index) > len(idx):
            idx = df.index
    if idx is None:
        return {"timeseries": pd.DataFrame(), "trades": pd.DataFrame(), "perf": pd.DataFrame()}

    symbols = sorted(prices_by_symbol.keys())
    open_px = {s: prices_by_symbol[s]["open"].reindex(idx).ffill() for s in symbols}
    close_px = {s: prices_by_symbol[s]["close"].reindex(idx).ffill() for s in symbols}
    high_px = {s: prices_by_symbol[s].get("high", prices_by_symbol[s]["close"]).reindex(idx).ffill() for s in symbols}
    low_px = {s: prices_by_symbol[s].get("low", prices_by_symbol[s]["close"]).reindex(idx).ffill() for s in symbols}

    signal = {s: signals_by_symbol.get(s, pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0) for s in symbols}

    bar_vol = {s: close_px[s].pct_change().rolling(20).std().fillna(0.0) for s in symbols}
    atr = {s: _safe_atr(high_px[s], low_px[s], close_px[s], window=14) for s in symbols}

    weights = pd.DataFrame(0.0, index=idx, columns=symbols)
    portfolio_ret = pd.Series(0.0, index=idx)

    trades_records: list[dict] = []

    target_sigma = target_bar_sigma(cfg.risk.annual_vol_target, cfg.interval)

    prev_weights = np.zeros(len(symbols))
    entry_close = {s: np.nan for s in symbols}

    for t in range(len(idx) - 1):
        ts = idx[t]
        ts_next = idx[t + 1]

        # Base desired from signal
        desired = np.array([signal[s].iat[t] for s in symbols])

        # ATR stop: if in position and price has moved against by atr_stop_mult * ATR, exit
        for i, s in enumerate(symbols):
            if prev_weights[i] > 0:
                atr_val = float(atr[s].iloc[t]) if not np.isnan(atr[s].iloc[t]) else 0.0
                stop_level = float(entry_close[s]) - cfg.risk.atr_stop_mult * atr_val
                if float(close_px[s].iloc[t]) <= stop_level:
                    desired[i] = 0.0

        # Kelly cap multiplier
        desired = desired * max(0.0, min(cfg.risk.kelly_cap, 1.0))

        # Per-trade cap by asset vol
        caps = np.array([per_trade_weight_cap(float(bar_vol[s].iat[t]), cfg.risk.per_trade_risk) for s in symbols])
        desired = np.minimum(desired, caps)

        # Apply position concentration limit
        desired = np.clip(desired, 0.0, cfg.risk.max_position_concentration)

        # Scale to portfolio vol target
        asset_vols = np.array([float(bar_vol[s].iat[t]) for s in symbols])
        current_proxy = portfolio_vol_proxy(desired, asset_vols)
        scale = 1.0 if current_proxy == 0 else min(1.0, target_sigma / current_proxy)
        sized = desired * scale

        # Orders
        orders = sized - prev_weights
        turnover = np.abs(orders).sum()

        # Fill at next open
        opx_next = np.array([float(open_px[s].loc[ts_next]) for s in symbols])
        cpx = np.array([float(close_px[s].loc[ts]) for s in symbols])
        ret_vec = (opx_next / cpx) - 1.0
        gross = float(np.dot(prev_weights, ret_vec))
        cost = turnover * cfg.cost.roundtrip_cost_rate
        pr = gross - cost
        portfolio_ret.iat[t + 1] = pr

        # Record trades for non-zero orders
        for i, s in enumerate(symbols):
            change = float(orders[i])
            if abs(change) > 1e-9:
                trades_records.append(
                    {
                        "datetime": ts_next,
                        "symbol": s,
                        "weight_prev": float(prev_weights[i]),
                        "weight_new": float(sized[i]),
                        "weight_change": change,
                        "fill_price": float(opx_next[i]),
                        "side": "BUY" if change > 0 else "SELL",
                        "turnover_abs": abs(change),
                        "est_cost": abs(change) * cfg.cost.roundtrip_cost_rate,
                    }
                )

        # Update entry price when initiating a new long
        for i, s in enumerate(symbols):
            if prev_weights[i] <= 1e-12 and sized[i] > 1e-12:
                entry_close[s] = float(close_px[s].loc[ts])
            if sized[i] <= 1e-12:
                entry_close[s] = np.nan

        prev_weights = sized
        weights.iloc[t + 1] = sized

    equity = (1.0 + portfolio_ret.fillna(0.0)).cumprod()
    dd = compute_drawdown(equity)

    # Enforce max drawdown: zero weights thereafter and recompute
    if dd.min() <= -cfg.risk.max_drawdown:
        breach_first = dd[dd <= -cfg.risk.max_drawdown].index[0]
        mask_after = weights.index >= breach_first
        weights.loc[mask_after, :] = 0.0
        portfolio_ret[:] = 0.0
        prev_weights = np.zeros(len(symbols))
        for t in range(len(idx) - 1):
            ts = idx[t]
            ts_next = idx[t + 1]
            orders = weights.iloc[t].to_numpy() - prev_weights
            turnover = np.abs(orders).sum()
            opx_next = np.array([float(open_px[s].loc[ts_next]) for s in symbols])
            cpx = np.array([float(close_px[s].loc[ts]) for s in symbols])
            ret_vec = (opx_next / cpx) - 1.0
            gross = float(np.dot(prev_weights, ret_vec))
            cost = turnover * cfg.cost.roundtrip_cost_rate
            portfolio_ret.iat[t + 1] = gross - cost
            prev_weights = weights.iloc[t + 1].to_numpy()
        equity = (1.0 + portfolio_ret.fillna(0.0)).cumprod()
        dd = compute_drawdown(equity)

    # Daily loss stop
    stop = daily_loss_stopped(portfolio_ret.fillna(0.0), threshold=cfg.risk.daily_loss_stop)
    if stop.any():
        dates = weights.index.tz_convert("UTC").date if weights.index.tzinfo else weights.index.date
        dfw = weights.copy()
        dfw["date"] = dates
        for d, grp in dfw.groupby("date"):
            m = stop.loc[grp.index]
            if m.any():
                first_idx = m.idxmax()
                mask = (dfw.index >= first_idx) & (dfw["date"] == d)
                weights.loc[mask, :] = 0.0
        portfolio_ret[:] = 0.0
        prev_weights = np.zeros(len(symbols))
        for t in range(len(idx) - 1):
            ts = idx[t]
            ts_next = idx[t + 1]
            orders = weights.iloc[t].to_numpy() - prev_weights
            turnover = np.abs(orders).sum()
            opx_next = np.array([float(open_px[s].loc[ts_next]) for s in symbols])
            cpx = np.array([float(close_px[s].loc[ts]) for s in symbols])
            ret_vec = (opx_next / cpx) - 1.0
            gross = float(np.dot(prev_weights, ret_vec))
            cost = turnover * cfg.cost.roundtrip_cost_rate
            portfolio_ret.iat[t + 1] = gross - cost
            prev_weights = weights.iloc[t + 1].to_numpy()
        equity = (1.0 + portfolio_ret.fillna(0.0)).cumprod()
        dd = compute_drawdown(equity)

    timeseries = pd.DataFrame({"portfolio_ret": portfolio_ret, "equity": equity, "drawdown": dd})
    timeseries.index.name = "datetime"

    trades = pd.DataFrame(trades_records)
    if not trades.empty:
        trades = trades.sort_values("datetime").set_index("datetime")

    # Perf summary
    ann_factor = 252.0 if cfg.interval == "1d" else (252.0 * (6.5 if cfg.interval == "60m" else 3.25))
    rets = timeseries["portfolio_ret"].fillna(0.0)
    vol = float(rets.std() * np.sqrt(ann_factor)) if len(rets) > 1 else 0.0
    ann_ret = float((1.0 + rets).prod() ** (ann_factor / max(1, len(rets))) - 1.0) if len(rets) > 0 else 0.0
    sharpe = float(ann_ret / vol) if vol > 0 else 0.0
    perf = pd.DataFrame(
        {
            "ann_return": [ann_ret],
            "ann_vol": [vol],
            "sharpe": [sharpe],
            "max_drawdown": [float(dd.min())],
            "num_trades": [int(len(trades))],
        }
    )

    return {"timeseries": timeseries, "trades": trades, "perf": perf}


__all__ = ["CostModel", "RiskLimits", "BacktestConfig", "backtest_signals"]