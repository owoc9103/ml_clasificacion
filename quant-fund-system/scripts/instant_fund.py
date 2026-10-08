"""Try instant funding via sandbox journal or direct cash injection."""
import os
import requests

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
    "Content-Type": "application/json",
}

base_url = "https://broker-api.sandbox.alpaca.markets"

# Get account
accounts_resp = requests.get(f"{base_url}/v1/accounts", headers=headers)
account_id = accounts_resp.json()[0]["id"]

print(f"Account ID: {account_id}\n")

# Method 1: Try sandbox instant fund endpoint (if exists)
print("Method 1: Trying instant fund endpoint...")
instant_data = {"amount": "50000"}
instant_resp = requests.post(
    f"{base_url}/v1/accounts/{account_id}/instant_fund",
    headers=headers,
    json=instant_data
)
if instant_resp.status_code in [200, 201]:
    print("✅ Instant fund successful!")
else:
    print(f"❌ Failed: {instant_resp.status_code} - {instant_resp.text[:100]}")

# Method 2: Try depositing via sandbox command
print("\nMethod 2: Trying sandbox deposit...")
deposit_resp = requests.post(
    f"{base_url}/v1/sandbox/accounts/{account_id}/deposit",
    headers=headers,
    json={"amount": "50000"}
)
if deposit_resp.status_code in [200, 201]:
    print("✅ Sandbox deposit successful!")
else:
    print(f"❌ Failed: {deposit_resp.status_code} - {deposit_resp.text[:100]}")

# Method 3: Check if we can just start trading (some sandboxes have unlimited virtual balance)
print("\nMethod 3: Testing unlimited virtual balance...")
test_order = {
    "symbol": "SPY",
    "notional": "10000",  # $10k worth
    "side": "buy",
    "type": "market",
    "time_in_force": "day"
}
order_resp = requests.post(
    f"{base_url}/v1/trading/accounts/{account_id}/orders",
    headers=headers,
    json=test_order
)
if order_resp.status_code in [200, 201]:
    print("✅ Order accepted with $0 balance!")
    print("   Sandbox allows unlimited virtual trading!")
    order = order_resp.json()
    # Cancel it
    requests.delete(
        f"{base_url}/v1/trading/accounts/{account_id}/orders/{order['id']}",
        headers=headers
    )
    print("   (Test order cancelled)")
else:
    print(f"❌ Order rejected: {order_resp.status_code}")

# Final balance check
print("\n" + "="*60)
account_resp = requests.get(
    f"{base_url}/v1/trading/accounts/{account_id}/account",
    headers=headers
)
account = account_resp.json()
print(f"\n💰 Final Balance:")
print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
