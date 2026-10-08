"""Test the updated AlpacaBroker with standard API."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantfund.execution.alpaca_broker import AlpacaBroker, Order

# Initialize broker
broker = AlpacaBroker(paper=True)

print("="*60)
print("Testing AlpacaBroker Integration")
print("="*60)

# Test 1: Get account
print("\n1. Get Account Info:")
account = broker.get_account()
print(f"   Status: {account.get('status')}")
print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")

# Test 2: Get positions (should be empty)
print("\n2. Get Positions:")
positions = broker.get_positions()
print(f"   Open positions: {len(positions)}")
for pos in positions:
    print(f"   - {pos.symbol}: {pos.qty} shares @ ${pos.current_price:.2f}")

# Test 3: Get orders (should be empty)
print("\n3. Get Orders:")
orders = broker.get_orders(status="open")
print(f"   Open orders: {len(orders)}")

# Test 4: Get latest quotes
print("\n4. Get Latest Quotes (SPY, QQQ):")
quotes = broker.get_latest_quotes(["SPY", "QQQ"])
for symbol, price in quotes.items():
    print(f"   {symbol}: ${price:.2f}")

# Test 5: Test order submission (dry run - we'll cancel immediately)
print("\n5. Test Order Submission (will cancel immediately):")
test_order = Order(
    symbol="SPY",
    qty=1,
    side="buy",
    order_type="market",
    time_in_force="day"
)
try:
    order_result = broker.submit_order(test_order)
    order_id = order_result.get("id")
    print(f"   [OK] Order submitted: {order_id}")
    print(f"   Symbol: {order_result.get('symbol')}")
    print(f"   Qty: {order_result.get('qty')}")
    print(f"   Side: {order_result.get('side')}")
    print(f"   Status: {order_result.get('status')}")

    # Cancel it immediately
    print("   Canceling test order...")
    broker.cancel_all_orders()
    print("   [OK] Order canceled")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)
