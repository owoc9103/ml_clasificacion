"""
Run live/paper trading with PDT-safe execution.

This script ensures BUY orders are sent as NOTIONAL (dollar amount)
so that Pattern Day Trader (PDT) daytrading_buying_power caps are respected.
SELL orders are sent with quantity (qty), because Alpaca only supports qty for sells.

Usage:
    python scripts/run_live.py --mode paper
    python scripts/run_live.py --mode live  # Real money!

Notes:
- We attempt to obtain target weights from your LiveTradingEngine. If the engine
  does not expose a "compute-only" API, we will call run_daily_strategy in a
  compute-only manner and ignore its order execution, then perform our own
  execution using PDT-safe logic.
- You can also pass --broker-only to skip the engine's execution path entirely
  and execute orders directly from provided target weights file.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import click
from dotenv import load_dotenv

# Project imports
from quantfund.utils.log import get_logger
from quantfund.execution.alpaca_broker import AlpacaBroker, Order

# Optional: your engine (for signals/weights)
try:
    from quantfund.execution.live_engine import LiveTradingEngine  # type: ignore
except Exception:  # pragma: no cover
    LiveTradingEngine = None  # engine is optional

# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logger = get_logger(__name__)

DOLLAR_EPS = float(os.getenv("QF_MIN_REBALANCE_DOLLARS", "10"))  # skip tiny trades
DTBP_BUFFER = float(os.getenv("QF_DTBP_BUFFER", "0.95"))         # 95% of cap


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_portfolio_value(broker: AlpacaBroker) -> float:
    """Return account portfolio value (fallback to cash + positions)."""
    acct = broker.get_account()
    pv = acct.get("portfolio_value")
    if pv is not None:
        return _safe_float(pv, 0.0)
    cash = _safe_float(acct.get("cash", 0.0), 0.0)
    mval = sum(abs(p.market_value) for p in broker.get_positions())
    return cash + mval


def get_prices(broker: AlpacaBroker, symbols: List[str]) -> Dict[str, float]:
    return broker.get_latest_quotes(symbols)


def rebalance_with_notional_buys(
    broker: AlpacaBroker,
    target_weights: Dict[str, float],
    buffer: float = DTBP_BUFFER,
) -> List[Dict]:
    """
    Execute rebalance where BUY legs are placed using NOTIONAL (PDT-safe)
    and SELL legs use QTY.

    target_weights: mapping of symbol -> target weight (0..1). Sum can be <= 1.
    buffer: extra safety factor to stay below DTBP cap in the broker (default from env).
    """
    results: List[Dict] = []

    symbols = list(target_weights.keys())
    prices = get_prices(broker, symbols)
    pv = get_portfolio_value(broker)

    # Current positions map
    current_pos = {p.symbol: p for p in broker.get_positions()}

    for sym in symbols:
        w = float(target_weights.get(sym, 0.0) or 0.0)
        px = _safe_float(prices.get(sym, 0.0), 0.0)
        if px <= 0:
            results.append({"symbol": sym, "skipped": True, "reason": "no_price"})
            continue

        target_notional = pv * w
        curr_notional = 0.0
        if sym in current_pos:
            curr_notional = float(current_pos[sym].qty) * px

        delta = target_notional - curr_notional

        # Skip tiny changes
        if abs(delta) < DOLLAR_EPS:
            results.append({"symbol": sym, "skipped": True, "reason": "small_delta"})
            continue

        if delta > 0:
            # BUY — PDT safe via NOTIONAL. Broker will cap to min(DTBP, cash)*buffer
            buy_notional = round(delta, 2)
            order = Order(
                symbol=sym,
                qty=None,
                side="buy",
                order_type="market",
                time_in_force="day",
                notional=buy_notional,
            )
            try:
                resp = broker.submit_order(order)
                results.append({
                    "symbol": sym,
                    "action": "BUY",
                    "notional": buy_notional,
                    "resp": resp,
                })
            except Exception as e:  # noqa: BLE001
                results.append({
                    "symbol": sym,
                    "action": "BUY",
                    "notional": buy_notional,
                    "error": str(e),
                })
        else:
            # SELL — qty only (notional unsupported for sells)
            sell_notional = abs(delta)
            qty = max(1, math.floor(sell_notional / px))  # min 1 share/contract
            if qty <= 0:
                results.append({"symbol": sym, "skipped": True, "reason": "qty_calc_zero"})
                continue
            order = Order(
                symbol=sym,
                qty=qty,
                side="sell",
                order_type="market",
                time_in_force="day",
            )
            try:
                resp = broker.submit_order(order)
                results.append({
                    "symbol": sym,
                    "action": "SELL",
                    "qty": qty,
                    "resp": resp,
                })
            except Exception as e:  # noqa: BLE001
                results.append({
                    "symbol": sym,
                    "action": "SELL",
                    "qty": qty,
                    "error": str(e),
                })

    return results


@click.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default="paper", help="Trading mode")
@click.option("--config", default="configs/live_trading.yaml", help="Config file path")
@click.option(
    "--symbols",
    default="SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLY,XLP,XLV,TLT,HYG",
    help="Comma-separated symbols",
)
@click.option("--dry-run", is_flag=True, help="Compute signals but don't execute orders")
@click.option(
    "--broker-only",
    is_flag=True,
    help="Skip engine execution and only use broker-side PDT-safe rebalancer (requires weights file)",
)
@click.option(
    "--weights-file",
    default="",
    help="Optional JSON file containing target_weights mapping {symbol: weight}",
)
@click.option(
    "--dtbp-buffer",
    default=None,
    help="Optional override for PDT DTBP safety buffer (e.g., 0.95)",
)
@click.option(
    "--log-summary/--no-log-summary",
    default=True,
    help="Print human-readable summary at the end",
)
def main(
    mode: str,
    config: str,
    symbols: str,
    dry_run: bool,
    broker_only: bool,
    weights_file: str,
    dtbp_buffer: str | None,
    log_summary: bool,
):
    """Run daily trading strategy with PDT-safe execution."""

    if mode == "live":
        click.confirm("\n⚠️  You are about to trade with REAL MONEY. Are you sure?", abort=True)

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    logger.info("Starting live trading in %s mode", mode)
    logger.info("Symbols (%d): %s", len(symbol_list), symbol_list)

    # Override buffer if provided
    global DTBP_BUFFER
    if dtbp_buffer is not None:
        try:
            DTBP_BUFFER = float(dtbp_buffer)
        except Exception:
            logger.warning("Invalid --dtbp-buffer value '%s' — keeping default %.2f", dtbp_buffer, DTBP_BUFFER)

    result: Dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "account_value": None,
        "orders": [],
        "target_weights": {},
        "engine": None,
    }

    broker = AlpacaBroker(paper=(mode == "paper"))

    # 1) Get/compute target weights
    target_weights: Dict[str, float] = {}

    # Option A: read from file when broker-only
    if broker_only and weights_file:
        with open(weights_file, "r", encoding="utf-8") as f:
            target_weights = json.load(f)
        logger.info("Loaded target_weights from %s", weights_file)
    else:
        # Option B: use engine if available
        if LiveTradingEngine is None:
            logger.warning("LiveTradingEngine not available; you must provide --broker-only --weights-file.")
        else:
            engine = LiveTradingEngine(config_path=config)
            # Call run_daily_strategy to get target weights
            weights = None
            try:
                engine_result = engine.run_daily_strategy(symbol_list, execute=False)
                # Some engines include weights in result
                if isinstance(engine_result, dict) and "target_weights" in engine_result:
                    weights = engine_result["target_weights"]
                    logger.info("Obtained target weights from run_daily_strategy() result.")
                else:
                    logger.warning("Engine did not return target_weights; you may need --broker-only mode.")
            except Exception as e:  # noqa: BLE001
                logger.error("Engine execution failed: %s", e, exc_info=True)

            if isinstance(weights, dict):
                # Normalize to symbols provided (ignore unknown keys)
                target_weights = {s: float(weights.get(s, 0.0) or 0.0) for s in symbol_list}

    result["target_weights"] = target_weights

    # 2) Execute PDT-safe rebalance (unless dry-run)
    if dry_run:
        logger.info("DRY RUN: skipping order execution. Showing computed weights only.")
        acct_val = get_portfolio_value(broker)
        result["account_value"] = acct_val
        print(json.dumps(result, indent=2))
        if log_summary:
            _print_summary(result)
        sys.exit(0)

    if not target_weights:
        logger.error("No target weights available; aborting execution.")
        print(json.dumps(result, indent=2))
        sys.exit(2)

    # Execute notional-buy rebalance
    results = rebalance_with_notional_buys(broker, target_weights, buffer=DTBP_BUFFER)

    # Build output payload
    acct_val = get_portfolio_value(broker)
    result["account_value"] = acct_val
    result["orders"] = results

    print(json.dumps(result, indent=2))

    if log_summary:
        _print_summary(result)


def _print_summary(result: Dict):
    print("\n" + "=" * 60)
    print(f"Timestamp: {result.get('timestamp')}")
    av = result.get("account_value")
    if av is not None:
        print(f"Account Value: ${av:,.2f}")
    orders = result.get("orders", [])
    print(f"Orders Executed: {len(orders)}")
    tw = result.get("target_weights", {})
    print(f"Target Positions: {len([w for w in tw.values() if (w or 0) > 0])}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        logger.error("Fatal error in run_live.py: %s", e, exc_info=True)
        sys.exit(1)
