# Quick Start: Deploy to AWS EC2 in 10 Minutes

The fastest way to get your trading bot running in the cloud.

## Prerequisites

- AWS account
- SSH client (built-in on Windows 11, Mac, Linux)
- Your Alpaca API keys

## Step 1: Launch EC2 Instance (3 minutes)

### Via AWS Console:

1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Click **"Launch Instance"**
3. Configure:
   - **Name:** `quantfund-trader`
   - **OS:** Ubuntu Server 22.04 LTS
   - **Instance type:** t3.micro (free tier) or t3.small ($0.02/hour)
   - **Key pair:** Create new or select existing
   - **Network:** Allow SSH from your IP only
4. Click **"Launch Instance"**
5. Wait 1-2 minutes for instance to start
6. Note the **Public IPv4 address**

## Step 2: Connect to Instance (1 minute)

```bash
# Replace with your key file and IP address
ssh -i "your-key.pem" ubuntu@<YOUR-VM-IP>
```

If you see "Permission denied", run:
```bash
chmod 400 your-key.pem
```

## Step 3: Run Auto-Setup (4 minutes)

Copy and paste this entire block:

```bash
# Install dependencies
sudo apt-get update && sudo apt-get install -y git curl

# Clone repository (replace with your repo URL if you forked it)
cd ~
git clone https://github.com/your-username/quant-fund-mvp.git quantfund
cd quantfund

# Run setup script
chmod +x deploy/aws_ec2_setup.sh
./deploy/aws_ec2_setup.sh

# Re-login for docker permissions
exit
```

Then SSH back in:
```bash
ssh -i "your-key.pem" ubuntu@<YOUR-VM-IP>
cd quantfund
```

## Step 4: Configure API Keys (1 minute)

```bash
# Create .env file
cat > .env << EOF
ALPACA_API_KEY=PKDZDT1U3X5RB1FWWXCB
ALPACA_SECRET_KEY=h0H9chxb2hiD8YE5VcN0eBN1K1pKY5SHg6UA6JEo
EOF
```

**Replace with your actual API keys!**

## Step 5: Start the Bot (1 minute)

```bash
# Build and start container
docker-compose build
docker-compose up -d

# Set up automated daily trading (9:50 AM ET)
chmod +x deploy/setup_cron.sh
./deploy/setup_cron.sh

# Test it works
docker-compose exec trading-bot python scripts/run_live.py --mode paper
```

You should see:
```
2025-10-01 20:54:22 | INFO | Strategy execution complete
Account Value: $99,993.03
Orders Generated: X
```

## ✅ Done! Your Bot is Running

### What happens now:

- **Automatically runs Monday-Friday at 9:50 AM ET**
- **Logs all trades** to `logs/live/`
- **Rebalances portfolio** based on strategy signals
- **Survives reboots** (auto-restarts)

### View logs:
```bash
tail -f logs/live/trading_*.log
```

### Monitor trades:
Visit [Alpaca Dashboard](https://app.alpaca.markets)

### Stop/Start:
```bash
docker-compose stop   # Stop trading
docker-compose start  # Resume trading
```

## Cost Breakdown

- **t3.micro (free tier):** $0/month for first 12 months, then ~$7/month
- **t3.small (faster):** ~$15/month
- **Data transfer:** ~$1/month

**Total: ~$0-16/month**

## Enable Slack Notifications (Optional)

1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Edit notification settings:
```bash
nano scripts/run_scheduled.sh
```
3. Uncomment and add your webhook URL in the error notification section

## Troubleshooting

**"Permission denied" on SSH:**
```bash
chmod 400 your-key.pem
```

**Container won't start:**
```bash
docker-compose logs
```

**Cron not running:**
```bash
crontab -l  # Should show the scheduled job
./scripts/run_scheduled.sh  # Test manually
```

**Need help?**
Check the full deployment guide: `deploy/DEPLOYMENT.md`

## Next Steps

1. ✅ Monitor for 1 week - Check logs daily
2. ✅ Review performance vs backtest
3. ✅ Add Slack notifications
4. ✅ Consider switching to live trading (⚠️ real money!)

Happy trading! 🚀📈
