import os
import sys
import math
import json
from glob import glob
from typing import Optional, Dict, List, cast

import click
import pandas as pd
from joblib import load

from quantfund.strategies.filters import compute_regimes
from quantfund.strategies.momo import momo_signals
from quantfund.strategies.meanrev import meanrev_signals
from quantfund.strategies.vol_regime import vol_regime_signals
from quantfund.backtest.engine import BacktestConfig, CostModel, RiskLimits, backtest_signals


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _load_partitioned_features(features_dir: str, interval: str) -> dict[str, pd.DataFrame]:
    base = os.path.join(features_dir, f"interval={interval}")
    paths = sorted(glob(os.path.join(base, "symbol=*", "data.parquet")))
    out = {}
    for p in paths:
        sym = os.path.basename(os.path.dirname(p)).split("=", 1)[-1]
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        out[sym] = df.sort_index()
    return out


def _apply_date_filter(by_symbol: Dict[str, pd.DataFrame], start: Optional[str], end: Optional[str]) -> Dict[str, pd.DataFrame]:
    if start is None and end is None:
        return by_symbol
    start_ts = pd.Timestamp(start).tz_localize("UTC") if start else None
    end_ts = pd.Timestamp(end).tz_localize("UTC") if end else None
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in by_symbol.items():
        cur = df
        # Normalize index to UTC DatetimeIndex to avoid tz-naive/aware issues
        cur = cur.copy()
        cur.index = pd.to_datetime(cur.index, utc=True)
        if start_ts is not None:
            cur = cur.loc[cur.index >= start_ts]
        if end_ts is not None:
            cur = cur.loc[cur.index <= end_ts]
        out[sym] = cur
    return out


@click.command()
@click.option("--features", "features_dir", default="data/datasets", show_default=True)
@click.option("--interval", "interval", default="60m", show_default=True)
@click.option("--models", "models_dir", default="artifacts", show_default=True)
@click.option("--use_model", is_flag=True, help="Load trained model probabilities if available")
@click.option("--start", "start", default=None, help="Filter start date (YYYY-MM-DD)")
@click.option("--end", "end", default=None, help="Filter end date (YYYY-MM-DD)")
@click.option("--strategies", "strategies", multiple=True, type=click.Choice(["momo", "meanrev"]))
@click.option("--commission", default=2.0, show_default=True, help="Commission bps")
@click.option("--slippage", default=1.0, show_default=True, help="Slippage bps")
@click.option("--vol_target", default=0.10, show_default=True, help="Annual vol target")
@click.option("--max_dd", default=0.12, show_default=True, help="Max drawdown")
@click.option("--per_trade", default=0.005, show_default=True, help="Per-trade risk cap")
@click.option("--daily_stop", default=0.01, show_default=True, help="Daily loss stop")
@click.option("--atr_mult", default=3.0, show_default=True, help="ATR stop multiple")
@click.option("--kelly_cap", default=0.15, show_default=True, help="Capped Kelly factor (0.1–0.2)")
@click.option("--use_regime_filter", is_flag=True, help="Apply regime filter (trend/vol/credit/VIX)")
@click.option("--momo_weight", default=0.5, show_default=True, help="Momentum strategy weight (0-1)")
@click.option("--vol_scale", is_flag=True, help="Scale position size by inverse volatility")
@click.option("--use_vol_regime", is_flag=True, help="Use ML volatility regime predictions to scale positions")
@click.option("--out_dir", default="reports", show_default=True)
def main(features_dir: str, interval: str, models_dir: str, use_model: bool, start: Optional[str], end: Optional[str], strategies: List[str], commission: float, slippage: float, vol_target: float, max_dd: float, per_trade: float, daily_stop: float, atr_mult: float, kelly_cap: float, use_regime_filter: bool, momo_weight: float, vol_scale: bool, use_vol_regime: bool, out_dir: str):
    by_symbol = _load_partitioned_features(features_dir, interval)
    if not by_symbol:
        click.echo("No features found.", err=True)
        sys.exit(2)

    by_symbol = _apply_date_filter(by_symbol, start, end)

    # Determine selected strategies
    selected = list(strategies) if strategies else ["momo", "meanrev"]

    # Compute regimes using SPY/HYG/TLT/^VIX from the filtered frames
    regimes = compute_regimes({k: v for k, v in by_symbol.items() if k in {"SPY", "HYG", "TLT", "^VIX"}})

    # Filter to tradeable symbols (exclude VIX and regime-only symbols)
    tradeable_symbols = {k: v for k, v in by_symbol.items() if k not in {"^VIX"}}

    # Load trained model and feature metadata
    model_path = os.path.join(models_dir, interval, "model_all.joblib")
    meta_path = os.path.join(models_dir, interval, "feature_meta.json")
    model = None
    feature_cols = None

    if use_model and os.path.exists(model_path) and os.path.exists(meta_path):
        model = load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
            feature_cols = meta["features"]

    proba_by_symbol = {}
    for sym, sdf in tradeable_symbols.items():
        if model is not None and feature_cols is not None:
            # Use trained model predictions
            X = sdf[feature_cols].fillna(0.0).astype(float)
            p = pd.Series(model.predict_proba(X.values)[:, 1], index=sdf.index)
        else:
            # Fallback to simple heuristic
            z = sdf.get("ema20_z")
            p = (z.fillna(0.0).apply(_sigmoid) if z is not None else pd.Series(0.5, index=sdf.index))
        proba_by_symbol[sym] = p

    signals = {}
    for sym, sdf in tradeable_symbols.items():
        if use_regime_filter:
            ok_series: pd.Series = cast(pd.Series, regimes["ok"]).reindex(sdf.index).fillna(False)
        else:
            # No regime filter - always allow trading
            ok_series = pd.Series(True, index=sdf.index)

        parts = []
        weights = []
        if "momo" in selected:
            parts.append(momo_signals(sdf, proba_by_symbol[sym], ok_series))
            weights.append(momo_weight)
        if "meanrev" in selected:
            parts.append(meanrev_signals(sdf, ok_series))
            weights.append(1.0 - momo_weight)
        if not parts:
            sig = pd.Series(0.0, index=sdf.index)
        else:
            # Weighted combination
            total_weight = sum(weights)
            sig = sum(p * w for p, w in zip(parts, weights)) / total_weight if total_weight > 0 else pd.Series(0.0, index=sdf.index)

        # Apply volatility scaling if enabled
        if vol_scale:
            # Scale signal by inverse of volatility ratio (reduce in high vol)
            vol_ratio_term = sdf["vol_ratio_term"] if "vol_ratio_term" in sdf.columns else pd.Series(1.0, index=sdf.index)
            # When short-term vol > long-term vol (ratio > 1), scale down
            vol_scaler = 1.0 / (1.0 + vol_ratio_term.clip(0.5, 2.0))  # range [0.33, 0.67]
            sig = sig * vol_scaler

        # Apply ML volatility regime scaling if enabled
        if use_vol_regime and model is not None and feature_cols is not None:
            sig = vol_regime_signals(sdf, sig, proba_by_symbol[sym])

        signals[sym] = sig

    prices: Dict[str, pd.DataFrame] = {s: cast(pd.DataFrame, sdf.loc[:, ["open", "high", "low", "close"]]) for s, sdf in tradeable_symbols.items()}

    cfg = BacktestConfig(
        interval=interval,  # type: ignore
        cost=CostModel(commission_bps=commission, slippage_bps=slippage),
        risk=RiskLimits(
            annual_vol_target=vol_target,
            max_drawdown=max_dd,
            per_trade_risk=per_trade,
            daily_loss_stop=daily_stop,
            atr_stop_mult=atr_mult,
            kelly_cap=kelly_cap,
        ),
    )

    result = backtest_signals(prices, signals, cfg)
    os.makedirs(out_dir, exist_ok=True)
    suffix = interval
    if start or end:
        suffix = f"{interval}_{(start or 'NA')}_{(end or 'NA')}"
    result["timeseries"].to_parquet(os.path.join(out_dir, f"bt_timeseries_{suffix}.parquet"), compression="zstd")
    result["trades"].to_parquet(os.path.join(out_dir, f"bt_trades_{suffix}.parquet"), compression="zstd")
    result["perf"].to_parquet(os.path.join(out_dir, f"bt_perf_{suffix}.parquet"), compression="zstd")
    click.echo(result["perf"].to_string(index=False))


if __name__ == "__main__":
    main()