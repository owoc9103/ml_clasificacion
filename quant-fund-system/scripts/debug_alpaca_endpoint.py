"""Debug script to find the correct Alpaca endpoint."""
import os
import requests

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

endpoints_to_try = [
    ("Paper Trading API v2", "https://paper-api.alpaca.markets/v2/account"),
    ("Broker Sandbox v1 accounts", "https://broker-api.sandbox.alpaca.markets/v1/accounts"),
    ("Broker Sandbox v1 trading", "https://broker-api.sandbox.alpaca.markets/v1/trading/accounts"),
    ("Data Sandbox", "https://data.sandbox.alpaca.markets/v2/account"),
]

print("Testing Alpaca endpoints...\n")

for name, url in endpoints_to_try:
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"✅ SUCCESS: {name}")
            print(f"   URL: {url}")
            print(f"   Response: {response.json()}")
            print()
            break
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            print(f"   {response.text[:200]}")
            print()
    except Exception as e:
        print(f"❌ {name}: {e}")
        print()
