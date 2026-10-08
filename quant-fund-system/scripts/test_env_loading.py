"""Test if .env file is being loaded correctly."""
import os
from pathlib import Path
from dotenv import load_dotenv

print("Current working directory:", os.getcwd())
print()

# Test 1: Load from explicit path
env_path = Path(__file__).parent.parent / ".env"
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")
print()

# Load it
result = load_dotenv(dotenv_path=env_path)
print(f"load_dotenv() returned: {result}")
print()

# Check if variables are loaded
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

print(f"ALPACA_API_KEY: {api_key[:10] if api_key else 'NOT FOUND'}...")
print(f"ALPACA_SECRET_KEY: {secret_key[:10] if secret_key else 'NOT FOUND'}...")
print()

if api_key and secret_key:
    print("✅ Credentials loaded successfully!")
else:
    print("❌ Credentials NOT loaded")
