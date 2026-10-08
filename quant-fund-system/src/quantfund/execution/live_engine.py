"""
Live trading engine.

Workflow:
1. Load today's market data
2. Generate signals using trained model + technical strategies
3. Calculate target weights with risk controls
4. Compute orders (target - current)
5. Execute orders via broker
6. Log everything
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from joblib import load

from quantfund.execution.alpaca_broker import AlpacaBroker, Order
from quantfund.strategies.momo import momo_signals
from quantfund.strategies.meanrev import meanrev_signals
from quantfund.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class LiveConfig:
    """Live trading configuration."""
    interval: str
    momo_weight: float
    vol_target: float
    per_trade_risk: float
    kelly_cap: float
    max_position_size: float
    min_position_size: float
    broker: AlpacaBroker


class LiveTradingEngine:
    """Execute strategy in live/paper trading."""

    def __init__(self, config_path: str = "configs/live_trading.yaml"):
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Initialize broker
        paper = "paper" in cfg["broker"]["base_url"]
        self.broker = AlpacaBroker(paper=paper)

        # Store config
        self.config = LiveConfig(
            interval=cfg["strategy"]["interval"],
            momo_weight=cfg["strategy"]["momo_weight"],
            vol_target=cfg["risk"]["annual_vol_target"],
            per_trade_risk=cfg["risk"]["per_trade_risk"],
            kelly_cap=cfg["risk"]["kelly_cap"],
            max_position_size=cfg["limits"]["max_position_size"],
            min_position_size=cfg["limits"]["min_position_size"],
            broker=self.broker,
        )

        from quantfund.models.multi_train import resolve_live_model_path

        model_path, meta_path = resolve_live_model_path(self.config.interval)
        if model_path.exists():
            self.model = load(model_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                import json

                meta = json.load(f)
                self.feature_cols = meta["features"]
            logger.info(f"Loaded model from {model_path} (key={meta.get('model', 'legacy')})")
        else:
            self.model = None
            self.feature_cols = None
            logger.warning("No model found, using heuristic signals only")

        logger.info("Live trading engine initialized")

    def get_live_features(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest features for symbols.

        In production, this would:
        1. Fetch latest OHLCV from data provider
        2. Compute features using same pipeline as backtest
        3. Return feature DataFrames per symbol

        For now, we'll use the most recent data from our dataset.
        """
        features = {}
        for symbol in symbols:
            path = f"data/datasets/interval={self.config.interval}/symbol={symbol}/data.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path)
                # Get most recent row (today's features)
                if not df.empty:
                    features[symbol] = df.tail(1)

        return features

    def generate_signals(self, features: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate trading signals for each symbol."""
        signals = {}

        for symbol, df in features.items():
            if df.empty:
                signals[symbol] = 0.0
                continue

            # Generate momentum signal
            if self.model and self.feature_cols:
                from quantfund.models.registry import predict_positive_proba

                X = df[self.feature_cols].fillna(0.0)
                proba = float(predict_positive_proba(self.model, X)[0])
            else:
                # Fallback heuristic
                z = df.get("ema20_z")
                proba = 1.0 / (1.0 + np.exp(-z.iloc[0])) if z is not None else 0.5

            momo_sig = (proba - 0.45) * (1.0 / 0.55) if proba > 0.45 else 0.0

            # Generate mean reversion signal
            rsi2 = df.get("rsi_2", pd.Series([50.0])).iloc[0]
            rsi14 = df.get("rsi_14", pd.Series([50.0])).iloc[0]
            ema20_z = df.get("ema20_z", pd.Series([0.0])).iloc[0]

            rsi2_oversold = (10 - min(rsi2, 10)) / 10.0
            rsi14_oversold = (30 - min(rsi14, 30)) / 30.0
            oversold = (rsi2_oversold * 0.5 + rsi14_oversold * 0.5)

            meanrev_sig = oversold * 0.7 + max(0, -ema20_z / 3.0) * 0.3

            # Weighted combination
            signal = momo_sig * self.config.momo_weight + meanrev_sig * (1 - self.config.momo_weight)
            signals[symbol] = float(np.clip(signal, 0, 1))

        return signals

    def compute_target_weights(self, signals: Dict[str, float], account_value: float) -> Dict[str, float]:
        """Convert signals to target portfolio weights with risk controls."""
        # Simple approach: scale signals to weights
        total_signal = sum(signals.values())
        if total_signal == 0:
            return {s: 0.0 for s in signals}

        weights = {}
        for symbol, signal in signals.items():
            raw_weight = signal / total_signal

            # Apply position limits
            weight = np.clip(raw_weight, 0, self.config.max_position_size)

            # Filter out tiny positions
            if weight < self.config.min_position_size:
                weight = 0.0

            weights[symbol] = weight

        # Normalize to sum to 1.0 (or less if using cash buffer)
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total * 0.95 for s, w in weights.items()}  # 95% invested, 5% cash

        return weights

    def compute_orders(self, target_weights: Dict[str, float], account_value: float) -> List[Order]:
        """Compute orders needed to reach target weights."""
        # Get current positions
        positions = self.broker.get_positions()
        current_weights = {}
        for pos in positions:
            current_weights[pos.symbol] = pos.market_value / account_value

        # Get latest prices
        symbols = list(set(list(target_weights.keys()) + list(current_weights.keys())))
        prices = self.broker.get_latest_quotes(symbols)

        orders = []
        for symbol in symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_w = current_weights.get(symbol, 0.0)
            delta_w = target_w - current_w

            # Convert weight change to dollar amount
            delta_value = delta_w * account_value

            # Skip tiny orders
            if abs(delta_value) < 100:  # Less than $100 change
                continue

            price = prices.get(symbol, 0)
            if price == 0:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            qty = int(delta_value / price)
            if qty == 0:
                continue

            side = "buy" if qty > 0 else "sell"
            orders.append(Order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                order_type="market",
                time_in_force="day",
            ))

        return orders

    def execute_orders(self, orders: List[Order]) -> List[dict]:
        """Execute orders via broker."""
        results = []
        for order in orders:
            try:
                result = self.broker.submit_order(order)
                results.append(result)
                if result.get("skipped"):
                    logger.info(
                        "Order skipped: %s %s (%s)",
                        order.side,
                        order.symbol,
                        result.get("reason"),
                    )
                elif result.get("id"):
                    logger.info("Order submitted: %s %s id=%s", order.side, order.symbol, result.get("id"))
                else:
                    logger.info("Order result: %s %s -> %s", order.side, order.symbol, result)
            except Exception as e:
                logger.error(f"Order failed: {order.side} {order.qty} {order.symbol} - {e}")
                results.append({
                    "error": str(e),
                    "order": {"symbol": order.symbol, "side": order.side, "qty": order.qty}
                })

        return results

    def run_daily_strategy(self, symbols: List[str], *, execute: bool = True) -> dict:
        """Main entry point: run strategy for today."""
        logger.info(f"Running daily strategy for {len(symbols)} symbols")

        # Get account info
        account = self.broker.get_account()
        account_value = float(account["equity"])
        logger.info(f"Account value: ${account_value:,.2f}")

        # Get features
        features = self.get_live_features(symbols)
        logger.info(f"Loaded features for {len(features)} symbols")

        # Generate signals
        signals = self.generate_signals(features)
        logger.info(f"Generated signals: {signals}")

        # Compute target weights
        target_weights = self.compute_target_weights(signals, account_value)
        logger.info(f"Target weights: {target_weights}")

        # Compute orders
        orders = self.compute_orders(target_weights, account_value)
        logger.info(f"Computed {len(orders)} orders")

        results: List[dict] = []
        if execute:
            results = self.execute_orders(orders)
        else:
            logger.info("Skipping order execution (compute-only mode)")

        return {
            "timestamp": datetime.now().isoformat(),
            "account_value": account_value,
            "signals": signals,
            "target_weights": target_weights,
            "orders": [{"symbol": o.symbol, "side": o.side, "qty": o.qty} for o in orders],
            "results": results,
        }


__all__ = ["LiveTradingEngine", "LiveConfig"]
