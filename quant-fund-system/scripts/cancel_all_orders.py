"""Cancel all orders and close all positions to reset paper account."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantfund.execution.alpaca_broker import AlpacaBroker

broker = AlpacaBroker(paper=True)

print("Canceling all open orders...")
canceled = broker.cancel_all_orders()
print(f"Canceled {len(canceled)} orders")

print("\nClosing all positions...")
result = broker.close_all_positions()
print("All positions closed")

# Check final state
account = broker.get_account()
positions = broker.get_positions()

print("\n" + "="*60)
print("Account Reset Complete")
print("="*60)
print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
print(f"Open Positions: {len(positions)}")
