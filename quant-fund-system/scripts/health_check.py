"""
Health check script for monitoring system status.
Run this periodically to ensure everything is working.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def check_credentials():
    """Check if API credentials are configured."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret:
        return False, "API credentials not found in .env"

    if len(api_key) < 10 or len(secret) < 10:
        return False, "API credentials appear invalid"

    return True, "Credentials OK"


def check_broker_connection():
    """Check if we can connect to Alpaca."""
    try:
        from quantfund.execution.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(paper=True)
        account = broker.get_account()

        status = account.get("status")
        cash = float(account.get("cash", 0))

        if status != "ACTIVE":
            return False, f"Account status: {status}"

        return True, f"Connected (Cash: ${cash:,.2f})"
    except Exception as e:
        return False, f"Connection failed: {str(e)[:100]}"


def check_model_files():
    """Check if trained model exists."""
    model_path = Path(__file__).parent.parent / "artifacts" / "1d" / "model_all.joblib"
    meta_path = Path(__file__).parent.parent / "artifacts" / "1d" / "feature_meta.json"

    if not model_path.exists():
        return False, "Model file not found"

    if not meta_path.exists():
        return False, "Feature metadata not found"

    # Check model age
    age_days = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).days

    return True, f"Model OK (age: {age_days} days)"


def check_recent_logs():
    """Check if system has run recently."""
    log_dir = Path(__file__).parent.parent / "logs" / "live"

    if not log_dir.exists():
        return False, "Log directory not found"

    # Find most recent log
    log_files = list(log_dir.glob("trading_*.log"))

    if not log_files:
        return False, "No log files found"

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    age = datetime.now() - datetime.fromtimestamp(latest_log.stat().st_mtime)

    # Check if run in last 48 hours (should run daily on weekdays)
    if age.total_seconds() > 48 * 3600:
        return False, f"Last run: {age.days} days ago"

    # Check for errors in latest log
    with open(latest_log) as f:
        content = f.read()
        if "ERROR" in content:
            error_lines = [line for line in content.split("\n") if "ERROR" in line]
            return False, f"Errors in latest log: {error_lines[0][:100]}"

    return True, f"Last run: {age.seconds // 3600}h ago"


def check_disk_space():
    """Check if sufficient disk space available."""
    import shutil

    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    percent_free = (free / total) * 100

    if percent_free < 10:
        return False, f"Low disk space: {free_gb}GB free ({percent_free:.1f}%)"

    return True, f"Disk space OK: {free_gb}GB free"


def main():
    """Run all health checks."""
    print("=" * 60)
    print("QUANT FUND TRADING SYSTEM - HEALTH CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    checks = [
        ("Credentials", check_credentials),
        ("Broker Connection", check_broker_connection),
        ("Model Files", check_model_files),
        ("Recent Activity", check_recent_logs),
        ("Disk Space", check_disk_space),
    ]

    all_passed = True

    for name, check_func in checks:
        try:
            passed, message = check_func()
            status = "PASS" if passed else "FAIL"
            symbol = "✓" if passed else "✗"

            print(f"[{status}] {name}")
            print(f"      {message}")
            print()

            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[ERROR] {name}")
            print(f"      Exception: {str(e)[:100]}")
            print()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("STATUS: ALL CHECKS PASSED")
        print("System is healthy and operational.")
        return 0
    else:
        print("STATUS: SOME CHECKS FAILED")
        print("Review failures above and take corrective action.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
