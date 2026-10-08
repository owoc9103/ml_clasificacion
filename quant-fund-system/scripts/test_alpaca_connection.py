"""
Quick test script to verify Alpaca connection.

Run this after setting your API keys to make sure everything works.
"""
import os
import sys

def test_connection():
    """Test Alpaca API connection."""

    # Check environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ ERROR: API keys not found!")
        print("\nPlease set environment variables:")
        print("  $env:ALPACA_API_KEY='your_key'")
        print("  $env:ALPACA_SECRET_KEY='your_secret'")
        return False

    print("✅ API keys found")
    print(f"   API Key: {api_key[:8]}...")

    # Test connection
    try:
        from quantfund.execution.alpaca_broker import AlpacaBroker

        print("\n📡 Connecting to Alpaca (paper trading)...")
        broker = AlpacaBroker(paper=True)

        print("✅ Connected successfully!")

        # Get account info
        account = broker.get_account()

        print("\n💰 Account Information:")
        print(f"   Account Value: ${float(account['equity']):,.2f}")
        print(f"   Cash: ${float(account['cash']):,.2f}")
        print(f"   Buying Power: ${float(account['buying_power']):,.2f}")
        print(f"   Status: {account['status']}")

        # Test getting positions
        positions = broker.get_positions()
        print(f"\n📊 Current Positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.qty} shares @ ${pos.current_price:.2f}")
            print(f"      P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct*100:.2f}%)")

        # Test getting quotes
        print("\n📈 Testing market data...")
        quotes = broker.get_latest_quotes(["SPY", "QQQ"])
        for symbol, price in quotes.items():
            print(f"   {symbol}: ${price:.2f}")

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to run paper trading!")
        print("Next step: python scripts/run_live.py --mode paper --dry-run")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your API keys are correct (copy from Alpaca dashboard)")
        print("2. Make sure you're using PAPER trading keys (not live)")
        print("3. Check your internet connection")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
