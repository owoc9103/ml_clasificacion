"""
Regime filters computed from multi-symbol features.

- Trend: SPY uptrend via EMA50 > EMA200 AND (MACD>MACD_SIGNAL if avail) AND (r_20>0 if avail)
- Vol: realized vol(20) below its 126-bar rolling median
- Credit: HYG/TLT ratio SMA20 > SMA50
- VIX: if ^VIX available, close < 25

If any component is missing, it defaults to True (non-blocking).
"""
from __future__ import annotations

from typing import Dict
from typing import cast

import numpy as np
import pandas as pd

def _ema(series: pd.Series, span: int) -> pd.Series:
    res = series.ewm(span=span, adjust=False, min_periods=span).mean()
    return cast(pd.Series, res)

def compute_regimes(features_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = None
    for df in features_by_symbol.values():
        if idx is None or len(df.index) > len(idx):
            idx = df.index
    if idx is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))

    spy = features_by_symbol.get("SPY")
    hyg = features_by_symbol.get("HYG")
    tlt = features_by_symbol.get("TLT")
    vix = features_by_symbol.get("^VIX")

    trend = pd.Series(True, index=idx)
    vol = pd.Series(True, index=idx)
    credit = pd.Series(True, index=idx)
    vix_ok = pd.Series(True, index=idx)

    if spy is not None and not spy.empty:
        s = spy.reindex(idx).ffill()
        close = s.get("close")
        if close is not None:
            ema50 = _ema(close, 50)
            ema200 = _ema(close, 200)
            cond_trend = (ema50 > ema200)
            macd = s.get("macd")
            macd_signal = s.get("macd_signal")
            if macd is not None and macd_signal is not None:
                cond_trend = cond_trend & (macd > macd_signal)
            r20 = s.get("r_20")
            if r20 is not None:
                cond_trend = cond_trend & (r20 > 0)
            trend = cond_trend.fillna(True)
            ret1 = close.pct_change()
            vol20 = ret1.rolling(20).std()
            med126 = vol20.rolling(126, min_periods=20).median()
            vol = (vol20 < med126).fillna(True)

    if hyg is not None and tlt is not None and not hyg.empty and not tlt.empty:
        s_hyg = hyg.reindex(idx).ffill()
        s_tlt = tlt.reindex(idx).ffill()
        close_hyg = s_hyg.get("close")
        close_tlt = s_tlt.get("close")
        if close_hyg is not None and close_tlt is not None:
            ratio = (close_hyg / close_tlt).replace([np.inf, -np.inf], np.nan).ffill()
            sma20 = ratio.rolling(20, min_periods=5).mean()
            sma50 = ratio.rolling(50, min_periods=10).mean()
            credit = (sma20 > sma50).fillna(True)

    if vix is not None and not vix.empty:
        s_vix = vix.reindex(idx).ffill()
        if "close" in s_vix:
            vix_ok = (s_vix["close"] < 25).fillna(True)

    ok = trend & vol & credit & vix_ok
    out = pd.DataFrame({"trend": trend.astype(bool), "vol": vol.astype(bool), "credit": credit.astype(bool), "vix_ok": vix_ok.astype(bool), "ok": ok.astype(bool)}, index=idx)
    out.index.name = "datetime"
    return out


__all__ = ["compute_regimes"]
