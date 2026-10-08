"""Test if .env file is being loaded correctly."""
import os
from pathlib import Path
from dotenv import load_dotenv

print("BEFORE load_dotenv:")
print(f"  ALPACA_API_KEY from env: {os.getenv('ALPACA_API_KEY')}")
print()

# Load .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)  # override=True forces overwrite

print("AFTER load_dotenv:")
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
print(f"  ALPACA_API_KEY: {api_key}")
print(f"  ALPACA_SECRET_KEY: {secret_key}")
