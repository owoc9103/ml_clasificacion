"""Setup bank relationship and fund account."""
import os
import requests
import json

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
accounts = accounts_resp.json()
account_id = accounts[0]["id"]

print(f"Account ID: {account_id}\n")

# Step 1: Create ACH relationship
print("Step 1: Creating ACH bank relationship...")
bank_data = {
    "account_owner_name": "Paper Trader",
    "bank_account_type": "CHECKING",
    "bank_account_number": "123456789",
    "bank_routing_number": "121000248",  # Wells Fargo test routing
    "nickname": "Test Bank"
}

try:
    bank_resp = requests.post(
        f"{base_url}/v1/accounts/{account_id}/ach_relationships",
        headers=headers,
        json=bank_data
    )

    if bank_resp.status_code in [200, 201]:
        bank = bank_resp.json()
        bank_id = bank["id"]
        print(f"✅ Bank relationship created: {bank_id}")

        # Step 2: Transfer funds (bank relationship auto-approved in sandbox)
        print("\nStep 2: Transferring $100,000...")
        transfer_data = {
            "transfer_type": "ach",
            "relationship_id": bank_id,
            "amount": "100000",
            "direction": "INCOMING"
        }

        transfer_resp = requests.post(
            f"{base_url}/v1/accounts/{account_id}/transfers",
            headers=headers,
            json=transfer_data
        )

        if transfer_resp.status_code in [200, 201]:
            transfer = transfer_resp.json()
            print(f"✅ Transfer initiated: {transfer.get('id')}")
            print(f"   Amount: ${transfer.get('amount')}")
            print(f"   Status: {transfer.get('status')}")

            # Check balance
            print("\nStep 3: Checking new balance...")
            account_resp = requests.get(
                f"{base_url}/v1/trading/accounts/{account_id}/account",
                headers=headers
            )
            account = account_resp.json()
            print(f"💰 Account Balance:")
            print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
            print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        else:
            print(f"❌ Transfer failed: {transfer_resp.status_code}")
            print(transfer_resp.text)
    else:
        print(f"❌ Bank creation failed: {bank_resp.status_code}")
        print(bank_resp.text)

except Exception as e:
    print(f"❌ Error: {e}")
