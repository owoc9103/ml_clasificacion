"""Transfer funds using existing bank relationship."""
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
accounts = accounts_resp.json()
account_id = accounts[0]["id"]

print(f"Account ID: {account_id}\n")

# Get existing bank relationships
print("Getting existing bank relationships...")
bank_resp = requests.get(f"{base_url}/v1/accounts/{account_id}/ach_relationships", headers=headers)

if bank_resp.status_code == 200:
    banks = bank_resp.json()
    if banks:
        bank_id = banks[0]["id"]
        print(f"✅ Found bank relationship: {bank_id}")

        # Transfer funds (daily limit is $50k)
        print("\nTransferring $50,000...")
        transfer_data = {
            "transfer_type": "ach",
            "relationship_id": bank_id,
            "amount": "50000",
            "direction": "INCOMING"
        }

        transfer_resp = requests.post(
            f"{base_url}/v1/accounts/{account_id}/transfers",
            headers=headers,
            json=transfer_data
        )

        if transfer_resp.status_code in [200, 201]:
            transfer = transfer_resp.json()
            print(f"✅ Transfer initiated!")
            print(f"   Transfer ID: {transfer.get('id')}")
            print(f"   Amount: ${transfer.get('amount')}")
            print(f"   Status: {transfer.get('status')}")

            # Check balance
            print("\nChecking new balance...")
            account_resp = requests.get(
                f"{base_url}/v1/trading/accounts/{account_id}/account",
                headers=headers
            )
            account = account_resp.json()
            print(f"\n💰 Account Balance:")
            print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
            print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        else:
            print(f"❌ Transfer failed: {transfer_resp.status_code}")
            print(transfer_resp.text)
    else:
        print("❌ No bank relationships found")
else:
    print(f"❌ Failed to get bank relationships: {bank_resp.status_code}")
