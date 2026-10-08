"""
Setup a sandbox trading account via Alpaca Broker API.
"""
import os
import requests
import json
from datetime import datetime

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key,
    "Content-Type": "application/json",
}

base_url = "https://broker-api.sandbox.alpaca.markets"

print("Creating sandbox trading account...\n")

# Create a sandbox account
account_data = {
    "contact": {
        "email_address": "trader@example.com",
        "phone_number": "555-666-7788",
        "street_address": ["123 Trading St"],
        "city": "New York",
        "state": "NY",
        "postal_code": "10001",
        "country": "USA"
    },
    "identity": {
        "given_name": "Paper",
        "family_name": "Trader",
        "date_of_birth": "1990-01-01",
        "tax_id": "457-55-5462",
        "tax_id_type": "USA_SSN",
        "country_of_citizenship": "USA",
        "country_of_birth": "USA",
        "country_of_tax_residence": "USA",
        "funding_source": ["employment_income"]
    },
    "disclosures": {
        "is_control_person": False,
        "is_affiliated_exchange_or_finra": False,
        "is_politically_exposed": False,
        "immediate_family_exposed": False
    },
    "agreements": [
        {
            "agreement": "margin_agreement",
            "signed_at": datetime.now().isoformat() + "Z",
            "ip_address": "192.168.1.1"
        },
        {
            "agreement": "account_agreement",
            "signed_at": datetime.now().isoformat() + "Z",
            "ip_address": "192.168.1.1"
        },
        {
            "agreement": "customer_agreement",
            "signed_at": datetime.now().isoformat() + "Z",
            "ip_address": "192.168.1.1"
        }
    ]
}

try:
    response = requests.post(
        f"{base_url}/v1/accounts",
        headers=headers,
        json=account_data
    )

    if response.status_code in [200, 201]:
        account = response.json()
        print("✅ Account created successfully!")
        print(f"\nAccount ID: {account.get('id')}")
        print(f"Status: {account.get('status')}")
        print(f"\nAccount details:")
        print(json.dumps(account, indent=2))
    else:
        print(f"❌ Error: HTTP {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")
