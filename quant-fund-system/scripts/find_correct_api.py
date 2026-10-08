"""Find which Alpaca API your keys work with."""
import os
import requests

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
}

print(f"Testing API key: {api_key}\n")
print("=" * 60)

# Test 1: Standard Trading API (Paper)
print("\n1. Testing Standard Paper Trading API...")
try:
    resp = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ SUCCESS - Standard Paper Trading Account")
        print(f"   Account Value: ${float(data.get('equity', 0)):,.2f}")
        print(f"   Cash: ${float(data.get('cash', 0)):,.2f}")
        print("\n   USE THIS: Standard Trading API")
        print("   Endpoint: https://paper-api.alpaca.markets")
    else:
        print(f"   ❌ Failed: {resp.text[:100]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Standard Trading API (Live)
print("\n2. Testing Standard Live Trading API...")
try:
    resp = requests.get("https://api.alpaca.markets/v2/account", headers=headers)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ SUCCESS - Standard Live Trading Account")
        print(f"   Account Value: ${float(data.get('equity', 0)):,.2f}")
        print(f"   ⚠️  WARNING: This is REAL MONEY!")
        print("\n   USE THIS: Standard Trading API (Live)")
        print("   Endpoint: https://api.alpaca.markets")
    else:
        print(f"   ❌ Failed: {resp.text[:100]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Broker API
print("\n3. Testing Broker API...")
try:
    resp = requests.get("https://broker-api.sandbox.alpaca.markets/v1/accounts", headers=headers)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ SUCCESS - Broker API")
        print(f"   Accounts found: {len(data)}")
        if data:
            print(f"   This is a Broker/Institution account")
        else:
            print(f"   No sub-accounts created yet")
    else:
        print(f"   ❌ Failed: {resp.text[:100]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("\nConclusion:")
print("Run this script and see which one shows ✅ SUCCESS")
print("That's the API you should use!")
