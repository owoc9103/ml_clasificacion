"""
Alpaca broker integration for live/paper trading.

Handles:
- Account info
- Position management
- Order execution (with PDT/DTBP-safe sizing for BUYs)
- Market data (backup to primary feed)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from quantfund.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class Order:
    symbol: str
    qty: Optional[float]               # may be None for BUYs if notional is used
    side: str                          # "buy" or "sell"
    order_type: str                    # "market" or "limit"
    time_in_force: str                 # "day", "gtc", etc
    limit_price: Optional[float] = None
    notional: Optional[float] = None   # NEW: preferred for BUY sizing under PDT/DTBP


class AlpacaBroker:
    """Alpaca API client for trading."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")

        # Standard individual account API endpoints
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

        logger.info(f"Initialized Alpaca broker (paper={paper}, endpoint={self.base_url})")

    # ---------------------------
    # Low-level HTTP helper
    # ---------------------------
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make authenticated request to Alpaca API."""
        url = f"{self.base_url}{endpoint}"
        r = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
        if not r.ok:
            # Log full payload from Alpaca for easier debugging (403 messages, etc.)
            try:
                logger.error("Alpaca API error %s %s -> %s | body=%s",
                             method, endpoint, r.status_code, r.text)
            except Exception:
                logger.error("Alpaca API error %s %s -> %s", method, endpoint, r.status_code)
            r.raise_for_status()
        return r.json()

    # ---------------------------
    # Account & Market Data
    # ---------------------------
    def get_account(self) -> dict:
        """Get account information."""
        return self._request("GET", "/v2/account")

    def _get_buy_notional_cap(self) -> float:
        """
        Max notional for a BUY before the safety buffer.

        When daytrading_buying_power is set (>0), cap intraday buys (PDT).
        When DTBP is 0 (common outside RTH or on non-PDT paths), fall back to
        regular buying_power/cash so paper accounts are not stuck at cap=0.
        """
        acct = self.get_account()
        cash = float(acct.get("cash", 0) or 0.0)
        buying_power = float(acct.get("buying_power", 0) or 0.0)
        regt = float(acct.get("regt_buying_power", 0) or 0.0)
        dtbp = float(acct.get("daytrading_buying_power", 0) or 0.0)

        if dtbp > 0:
            if cash > 0:
                return max(0.0, min(dtbp, cash))
            return max(0.0, dtbp)

        fallback = buying_power if buying_power > 0 else max(regt, cash)
        if cash > 0 and fallback > 0:
            return max(0.0, min(fallback, cash))
        return max(0.0, fallback)

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest mid quotes for symbols using data API."""
        symbols_str = ",".join(symbols)
        url = f"{self.data_url}/v2/stocks/quotes/latest?symbols={symbols_str}"
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        quotes: Dict[str, float] = {}
        for symbol, quote in data.get("quotes", {}).items():
            # Use mid price (bid + ask) / 2 when available, fallback to last price if provided.
            bid = float(quote.get("bp", 0) or 0)
            ask = float(quote.get("ap", 0) or 0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            last = float(quote.get("p", 0) or 0)
            quotes[symbol] = mid if mid > 0 else last

        missing = [s for s in symbols if quotes.get(s, 0) <= 0]
        if missing:
            quotes.update(self._latest_trade_prices(missing))
        return quotes

    def _latest_trade_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fallback when bid/ask quotes are empty (e.g. after hours)."""
        out: Dict[str, float] = {}
        if not symbols:
            return out
        symbols_str = ",".join(symbols)
        url = f"{self.data_url}/v2/stocks/trades/latest?symbols={symbols_str}&feed=iex"
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            for symbol, trade in (r.json().get("trades") or {}).items():
                px = float(trade.get("p", 0) or 0)
                if px > 0:
                    out[symbol] = px
        except Exception as e:
            logger.warning("Latest trade fallback failed: %s", e)
        return out

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Convenience wrapper to fetch a single symbol's approximate last/mid price."""
        try:
            px = self.get_latest_quotes([symbol]).get(symbol, 0.0)
            return px if px > 0 else None
        except Exception as e:
            logger.warning("Failed to get last price for %s: %s", symbol, e)
            return None

    def get_bars(self, symbols: List[str], timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bars (backup data source).

        timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
        """
        symbols_str = ",".join(symbols)
        data = self._request("GET", "/v2/stocks/bars", params={
            "symbols": symbols_str,
            "timeframe": timeframe,
            "limit": limit,
        })

        dfs = []
        for symbol, bars in data.get("bars", {}).items():
            df = pd.DataFrame(bars)
            df["symbol"] = symbol
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined["t"] = pd.to_datetime(combined["t"])
        return combined

    # ---------------------------
    # Positions & Orders
    # ---------------------------
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        data = self._request("GET", "/v2/positions")
        positions = []
        for p in data:
            positions.append(Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
                unrealized_pl_pct=float(p.get("unrealized_plpc", 0) or 0),
            ))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        try:
            p = self._request("GET", f"/v2/positions/{symbol}")
            return Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
                unrealized_pl_pct=float(p.get("unrealized_plpc", 0) or 0),
            )
        except requests.HTTPError as e:
            if getattr(e, "response", None) and e.response.status_code == 404:
                return None
            raise

    def close_all_positions(self) -> dict:
        """Close all open positions."""
        return self._request("DELETE", "/v2/positions")

    # -------- PDT-safe BUY sizing helpers --------
    def _compute_desired_notional(self, order: Order) -> Optional[float]:
        """Infer desired notional from order.notional or (qty * last_price)."""
        if order.notional is not None:
            return float(order.notional)

        if order.qty is not None:
            px = self.get_last_price(order.symbol)
            if px and px > 0:
                return float(order.qty) * float(px)

        return None

    def _apply_dtbp_cap(self, desired_notional: float, buffer: float = 0.95) -> float:
        """
        Apply PDT daytrading_buying_power cap with a safety buffer.
        Returns the safe notional (could be 0).
        """
        cap = self._get_buy_notional_cap()
        safe = min(desired_notional, cap * buffer)
        safe_rounded = round(max(0.0, safe), 2)
        logger.info(
            "Buy notional cap check: desired=%.2f, cap=%.2f, safe=%.2f",
            desired_notional,
            cap,
            safe_rounded,
        )
        return safe_rounded

    def _post_order(self, payload: dict) -> dict:
        """Internal single attempt post with detailed logging."""
        logger.info("Submitting order payload: %s", payload)
        return self._request("POST", "/v2/orders", json=payload)

    def submit_order(self, order: Order) -> dict:
        """
        Submit an order with PDT/DTBP-safe handling for BUYs.

        BUY logic:
          - Prefer 'notional' sizing under PDT (Alpaca supports notional for MARKET buys).
          - If qty is provided, convert qty->notional via latest mid price.
          - Cap notional at min(daytrading_buying_power, cash) * 0.95.
          - Force 'market' if using notional (Alpaca limitation).

        SELL logic:
          - Use qty (notional isn't supported for sells).
          - If qty is None, attempt to use full position size.
        """
        side = order.side.lower()
        otype = order.order_type.lower()
        tif = order.time_in_force.lower()

        # Base payload
        payload: Dict[str, str] = {
            "symbol": order.symbol,
            "side": side,
            "type": otype,
            "time_in_force": tif,
        }

        # ---------- BUY: notional-sized with DTBP protection ----------
        if side == "buy":
            desired_notional = self._compute_desired_notional(order)
            if desired_notional is None or desired_notional <= 0:
                logger.warning("[%s] No notional/price inferred for BUY; skipping.", order.symbol)
                return {"skipped": True, "reason": "unknown_notional"}

            safe_notional = self._apply_dtbp_cap(desired_notional, buffer=0.95)
            if safe_notional <= 0:
                logger.warning("[%s] Skipping BUY — insufficient dtbp/cash.", order.symbol)
                return {"skipped": True, "reason": "insufficient_dtbp"}

            # Alpaca supports 'notional' with MARKET orders only; enforce if needed
            if otype != "market":
                logger.info("[%s] Forcing order_type=market to use notional sizing under PDT.", order.symbol)
                payload["type"] = "market"

            payload["notional"] = str(safe_notional)
            # Limit price shouldn't be sent with notional market; ignore if provided
            if order.limit_price:
                logger.info("[%s] Ignoring limit_price on notional BUY (market only).", order.symbol)

            # Primary attempt
            try:
                return self._post_order(payload)
            except requests.HTTPError as e:
                # If PDT error, retry with half notional once
                msg = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
                if "insufficient day trading buying power" in msg.lower():
                    retry_notional = round(safe_notional * 0.5, 2)
                    if retry_notional > 0:
                        logger.warning("[%s] PDT 403 — retrying with half notional: %.2f",
                                       order.symbol, retry_notional)
                        payload["notional"] = str(retry_notional)
                        return self._post_order(payload)
                # Re-raise if not handled
                raise

        # ---------- SELL: must use qty ----------
        else:
            qty = float(order.qty) if order.qty is not None else None
            if qty is None:
                pos = self.get_position(order.symbol)
                if pos is None or abs(pos.qty) <= 0:
                    logger.warning("[%s] No position to SELL; skipping.", order.symbol)
                    return {"skipped": True, "reason": "no_position"}
                qty = abs(float(pos.qty))
                logger.info("[%s] SELL qty inferred from position: %.4f", order.symbol, qty)

            payload["qty"] = str(abs(int(qty))) if qty >= 1 else str(max(qty, 0.0001))
            if order.limit_price and otype == "limit":
                payload["limit_price"] = str(float(order.limit_price))

            return self._post_order(payload)

    def get_orders(self, status: str = "open") -> List[dict]:
        """Get orders by status."""
        data = self._request("GET", "/v2/orders", params={"status": status})
        # If the API returns a dict, extract the list of orders; otherwise, return as is
        if isinstance(data, dict) and "orders" in data:
            return data["orders"]
        return data if isinstance(data, list) else []

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        return self._request("DELETE", "/v2/orders")


__all__ = ["AlpacaBroker", "Position", "Order"]
