"""Test standard Alpaca API connection."""
import os
import requests

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

base_url = "https://paper-api.alpaca.markets"

# Get account
print("Testing Standard Paper Trading API...")
print(f"API Key: {api_key}\n")

account_resp = requests.get(f"{base_url}/v2/account", headers=headers)
print(f"Status: {account_resp.status_code}")

if account_resp.status_code == 200:
    account = account_resp.json()
    print("\nSUCCESS - Standard Paper Trading Account")
    print("="*60)
    print(f"\nAccount Status: {account.get('status')}")
    print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
    print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
    print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
    print(f"Trading Blocked: {account.get('trading_blocked')}")
    print(f"Account Blocked: {account.get('account_blocked')}")

    if float(account.get('buying_power', 0)) > 0:
        print("\n[OK] Account is funded and ready to trade!")
    else:
        print("\n[WARNING] Account has no buying power")
else:
    print(f"\nFAILED: {account_resp.status_code}")
    print(account_resp.text)
