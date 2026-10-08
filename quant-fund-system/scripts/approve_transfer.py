"""Approve pending transfer in sandbox."""
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

# Get all transfers
transfers_resp = requests.get(f"{base_url}/v1/accounts/{account_id}/transfers", headers=headers)
transfers = transfers_resp.json()

print("Pending transfers:")
for t in transfers:
    if t.get("status") in ["QUEUED", "PENDING"]:
        transfer_id = t["id"]
        print(f"  Transfer ID: {transfer_id}")
        print(f"  Amount: ${t.get('amount')}")
        print(f"  Status: {t.get('status')}")

        # Try to approve it (sandbox only)
        print(f"\n  Attempting to approve...")
        approve_resp = requests.patch(
            f"{base_url}/v1/accounts/{account_id}/transfers/{transfer_id}",
            headers=headers,
            json={"status": "COMPLETE"}
        )

        if approve_resp.status_code in [200, 204]:
            print("  ✅ Transfer approved!")
        else:
            print(f"  Status: {approve_resp.status_code}")
            # Try alternate approval endpoint
            approve_resp2 = requests.post(
                f"{base_url}/v1/accounts/{account_id}/transfers/{transfer_id}/approve",
                headers=headers
            )
            if approve_resp2.status_code in [200, 204]:
                print("  ✅ Transfer approved (alternate endpoint)!")
            else:
                print(f"  ℹ️  Transfer may auto-approve in sandbox")

# Check final balance
print("\n" + "="*60)
account_resp = requests.get(
    f"{base_url}/v1/trading/accounts/{account_id}/account",
    headers=headers
)
account = account_resp.json()
print(f"\n💰 Current Account Balance:")
print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
print(f"   Status: {account.get('status')}")
