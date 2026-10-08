"""Check if account can trade without funding (some sandboxes allow this)."""
import os
import requests
import json

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

base_url = "https://broker-api.sandbox.alpaca.markets"

# Get account
accounts_resp = requests.get(f"{base_url}/v1/accounts", headers=headers)
accounts = accounts_resp.json()
account_id = accounts[0]["id"]

# Get full account details
account_resp = requests.get(f"{base_url}/v1/trading/accounts/{account_id}/account", headers=headers)
account = account_resp.json()

print("📊 Full Account Details:")
print(json.dumps(account, indent=2))

print("\n" + "="*60)
print("\nKey Info:")
print(f"Status: {account.get('status')}")
print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
print(f"Pattern Day Trader: {account.get('pattern_day_trader')}")
print(f"Trading Blocked: {account.get('trading_blocked')}")
print(f"Account Blocked: {account.get('account_blocked')}")

# Try a small test order to see if it works with $0
print("\n" + "="*60)
print("\n🧪 Testing if we can place orders with $0 balance...")

test_order = {
    "symbol": "SPY",
    "qty": 1,
    "side": "buy",
    "type": "market",
    "time_in_force": "day"
}

try:
    order_resp = requests.post(
        f"{base_url}/v1/trading/accounts/{account_id}/orders",
        headers=headers,
        json=test_order
    )

    if order_resp.status_code in [200, 201]:
        print("✅ Order accepted! Sandbox allows trading with $0 balance")
        print("   (Cancelling order...)")
        order = order_resp.json()
        requests.delete(
            f"{base_url}/v1/trading/accounts/{account_id}/orders/{order['id']}",
            headers=headers
        )
    else:
        print(f"❌ Order rejected: {order_resp.status_code}")
        print(f"   {order_resp.text}")
        print("\n   We need to fund the account first.")
except Exception as e:
    print(f"❌ Error: {e}")
