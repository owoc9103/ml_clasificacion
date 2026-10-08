"""Fund sandbox account with virtual money."""
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

# Get account ID
accounts_resp = requests.get(f"{base_url}/v1/accounts", headers=headers)
accounts = accounts_resp.json()

if not accounts:
    print("❌ No accounts found")
    exit(1)

account_id = accounts[0]["id"]
print(f"Account ID: {account_id}")

# Fund the account with $100,000 using journal entry (sandbox only)
fund_data = {
    "entry_type": "JNLC",  # Cash journal
    "from_account": account_id,
    "to_account": account_id,
    "amount": "100000",
    "description": "Initial funding for paper trading"
}

print(f"\n💰 Adding $100,000 to account...")

try:
    resp = requests.post(
        f"{base_url}/v1/journals",
        headers=headers,
        json=fund_data
    )

    if resp.status_code in [200, 201]:
        transfer = resp.json()
        print("✅ Funds added successfully!")
        print(f"   Transfer ID: {transfer.get('id')}")
        print(f"   Amount: ${transfer.get('amount')}")
        print(f"   Status: {transfer.get('status')}")

        # Check new balance
        account_resp = requests.get(
            f"{base_url}/v1/trading/accounts/{account_id}/account",
            headers=headers
        )
        account = account_resp.json()
        print(f"\n💵 New Account Balance:")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
    else:
        print(f"❌ Error: {resp.status_code}")
        print(resp.text)

except Exception as e:
    print(f"❌ Error: {e}")
