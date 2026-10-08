# Paper Trading Setup Guide

## Overview

This guide walks you through setting up paper trading with Alpaca to test the ETF strategy with fake money before going live.

## Prerequisites

- Python 3.11+
- All project dependencies installed (`pip install -r requirements.txt`)
- Trained model in `artifacts/1d/`

## Step 1: Sign Up for Alpaca

1. Go to https://alpaca.markets
2. Click "Sign Up" → Choose "Paper Trading" (free)
3. Complete registration
4. Verify email

## Step 2: Get API Keys

1. Log into Alpaca dashboard
2. Click "Generate API Keys" (or go to https://app.alpaca.markets/paper/dashboard/overview)
3. Copy your **API Key** and **Secret Key**
4. **IMPORTANT**: Store these securely - you'll need them

## Step 3: Set Environment Variables

### On Windows (PowerShell):
```powershell
$env:ALPACA_API_KEY="your_key_here"
$env:ALPACA_SECRET_KEY="your_secret_here"
```

### On Windows (CMD):
```cmd
set ALPACA_API_KEY=your_key_here
set ALPACA_SECRET_KEY=your_secret_here
```

### On Mac/Linux:
```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
```

### Permanent Setup (Recommended):

Create a `.env` file in the project root:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

## Step 4: Install Additional Dependencies

```bash
pip install requests pyyaml python-dotenv
```

## Step 5: Test Connection

```python
from quantfund.execution.alpaca_broker import AlpacaBroker

# Initialize broker
broker = AlpacaBroker(paper=True)

# Get account info
account = broker.get_account()
print(f"Account Value: ${float(account['equity']):,.2f}")
print(f"Buying Power: ${float(account['buying_power']):,.2f}")
```

## Step 6: Run Paper Trading (Dry Run First)

```bash
# Dry run - compute signals but don't execute
python scripts/run_live.py --mode paper --dry-run

# Actual paper trading (fake money)
python scripts/run_live.py --mode paper
```

## Step 7: Monitor Results

### Check positions:
```python
from quantfund.execution.alpaca_broker import AlpacaBroker

broker = AlpacaBroker(paper=True)
positions = broker.get_positions()

for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares @ ${pos.current_price:.2f}")
    print(f"  P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct*100:.2f}%)")
```

### View in Dashboard:
- Go to https://app.alpaca.markets/paper/dashboard/overview
- See positions, orders, account history

## Step 8: Schedule Daily Execution

### Using Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 9:50 AM (market open)
4. Action: Start a program
   - Program: `python`
   - Arguments: `C:\projects\quant-fund-mvp\scripts\run_live.py --mode paper`
   - Start in: `C:\projects\quant-fund-mvp`

### Using cron (Mac/Linux):
```bash
# Edit crontab
crontab -e

# Add line (runs at 9:50 AM ET on weekdays)
50 9 * * 1-5 cd /path/to/quant-fund-mvp && python scripts/run_live.py --mode paper
```

## Strategy Configuration

Edit `configs/live_trading.yaml` to adjust:

```yaml
strategy:
  momo_weight: 0.3        # 30% momentum, 70% mean reversion

risk:
  annual_vol_target: 0.20  # 20% target volatility
  per_trade_risk: 0.015    # 1.5% per trade
  kelly_cap: 0.30          # 30% Kelly cap

limits:
  max_position_size: 0.30  # Max 30% in any ETF
  max_num_positions: 12    # Max 12 concurrent positions
```

## Safety Checklist

Before running live trading:

- [ ] Backtest shows consistent Sharpe > 0.5 over 3+ years
- [ ] Paper trading runs successfully for 1+ month
- [ ] Paper trading Sharpe matches backtest (±0.2)
- [ ] Slippage and costs match expectations
- [ ] No unexpected errors in logs
- [ ] Position sizes are reasonable
- [ ] Drawdowns stay within limits

## Troubleshooting

### "API credentials not found"
- Check environment variables are set: `echo $ALPACA_API_KEY`
- Verify keys are correct (copy-paste from Alpaca dashboard)

### "Market is closed"
- Alpaca only trades during market hours: 9:30 AM - 4:00 PM ET
- Use `--dry-run` to test outside hours

### "Insufficient buying power"
- Paper account starts with $100,000
- Check you haven't exceeded this with position sizes

### "No model found"
- Run training first: `python scripts/train_model.py --start 2010-01-01 --end 2024-12-31 --interval 1d`
- Verify `artifacts/1d/model_all.joblib` exists

## Going Live (Real Money)

**ONLY** after 1+ month of successful paper trading:

1. Apply for live Alpaca account
2. Fund account (minimum $25k for PDT rule in US)
3. Get live API keys
4. Update `configs/live_trading.yaml`:
   ```yaml
   broker:
     base_url: "https://api.alpaca.markets"  # Live!
   ```
5. Set live API keys in environment
6. Run: `python scripts/run_live.py --mode live`

**Start small** - only 10-20% of capital for first 3 months.

## Support

- Alpaca Docs: https://alpaca.markets/docs
- Alpaca Community: https://forum.alpaca.markets
- Project Issues: https://github.com/your-repo/issues
