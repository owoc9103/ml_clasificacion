# Deployment Guide - AWS EC2

This guide will help you deploy the QuantFund trading system to your AWS EC2 instance.

## Prerequisites

- AWS EC2 instance running Ubuntu 22.04 (t3.small or larger)
- SSH key for EC2 access
- Alpaca API account (paper or live)
- Git repository with your code

## Instance Details (from your setup)

- Instance ID: `i-0c72554b9bdcd6ddf`
- Instance Name: `quant-fund-bot`
- Type: `t3.small`
- Public IP: `54.242.154.240`
- DNS: `ec2-54-242-154-240.compute-1.amazonaws.com`

## Step 1: SSH into EC2

```bash
ssh -i your-key.pem ubuntu@54.242.154.240
```

## Step 2: Run System Setup

```bash
# Download setup script
cat > aws_ec2_setup.sh << 'EOF'
#!/bin/bash
set -e

echo "Setting up Quant Fund Trading System..."

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Install Git
sudo apt-get install -y git

echo "Setup complete! Log out and back in for docker group changes to take effect."
EOF

chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh
```

**IMPORTANT:** Log out and back in after running the setup script:
```bash
exit
# Then SSH back in
ssh -i your-key.pem ubuntu@54.242.154.240
```

## Step 3: Clone Your Repository

```bash
mkdir -p ~/quantfund
cd ~/quantfund
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .
```

## Step 4: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env
```

Update `.env` with your Alpaca credentials:
```bash
ALPACA_API_KEY=your_actual_paper_key
ALPACA_SECRET_KEY=your_actual_paper_secret
ACCOUNT_TYPE=paper
ALPACA_ENDPOINT=https://paper-api.alpaca.markets/v2
```

## Step 5: Build and Start Docker Container

```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# Check logs
docker-compose logs -f
```

## Step 6: Verify System Health

```bash
# Run health check
docker-compose exec trading-bot python scripts/health_check.py

# Test manual run (dry-run)
docker-compose exec trading-bot python scripts/run_live.py --mode paper --dry-run
```

## Step 7: Setup Automated Trading (Cron)

```bash
# Make script executable
chmod +x scripts/run_scheduled.sh

# Setup cron job (runs Mon-Fri at 9:50 AM ET)
./deploy/setup_cron.sh

# Verify cron is set
crontab -l
```

## Step 8: Monitor Logs

```bash
# View live logs
tail -f logs/live/trading_*.log

# View most recent log
ls -lt logs/live/ | head -5

# Docker container logs
docker-compose logs -f trading-bot
```

## Testing the Deployment

### Test 1: Health Check
```bash
docker-compose exec trading-bot python scripts/health_check.py
```

Expected output:
- ✓ Credentials OK
- ✓ Broker Connection OK
- ✓ Model Files OK
- ✓ Disk Space OK

### Test 2: Dry Run
```bash
docker-compose exec trading-bot python scripts/run_live.py --mode paper --dry-run
```

This should:
1. Fetch latest market data
2. Generate trading signals
3. Calculate target positions
4. Show orders (but NOT execute)

### Test 3: Paper Trading
```bash
docker-compose exec trading-bot python scripts/run_live.py --mode paper
```

This will actually execute paper trades on Alpaca.

## Scheduled Trading

Once cron is setup, the system will automatically:
- Run Monday-Friday at 9:50 AM ET (14:50 UTC)
- Generate signals based on market open prices
- Execute trades via Alpaca paper account
- Log all activity to `logs/live/trading_YYYYMMDD_HHMMSS.log`

## Maintenance Commands

```bash
# Restart container
docker-compose restart

# Rebuild after code changes
git pull
docker-compose build
docker-compose up -d

# View running containers
docker ps

# Clean up old logs (manual)
find logs/live -name "trading_*.log" -mtime +30 -delete

# Stop trading
docker-compose down

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y
```

## Monitoring & Alerts

### Add Slack Notifications (Optional)

Edit `scripts/run_scheduled.sh` and uncomment the webhook section:
```bash
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"Trading run failed at $(date)\"}" \
  YOUR_SLACK_WEBHOOK_URL
```

### Check Alpaca Dashboard

- Paper: https://app.alpaca.markets/paper/dashboard/overview
- Live: https://app.alpaca.markets/live/dashboard/overview

## Troubleshooting

### Container won't start
```bash
docker-compose logs trading-bot
# Check for missing .env or API key issues
```

### API connection failed
```bash
# Test credentials
docker-compose exec trading-bot python -c "import os; from quantfund.execution.alpaca_broker import AlpacaBroker; broker = AlpacaBroker(paper=True); print(broker.get_account())"
```

### Model not found
```bash
# Check models exist
docker-compose exec trading-bot ls -la artifacts/1d/
```

### Cron not running
```bash
# Check crontab
crontab -l

# Check cron logs
grep CRON /var/log/syslog

# Test script manually
./scripts/run_scheduled.sh
```

## Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore` for a reason
2. **Use paper trading first** - Test thoroughly before live trading
3. **Rotate API keys regularly** - Change keys every 90 days
4. **Monitor logs daily** - Check for errors or unusual activity
5. **Set up CloudWatch alarms** - Monitor EC2 CPU/memory usage
6. **Regular backups** - Backup logs and trading records

## Switching to Live Trading

⚠️ **WARNING**: Live trading uses real money. Only switch after extensive paper trading.

```bash
# Stop container
docker-compose down

# Update .env with LIVE credentials
nano .env

# Update docker-compose.yml (change --mode paper to --mode live)
nano docker-compose.yml

# Rebuild and start
docker-compose build
docker-compose up -d
```

## Cost Estimation

- EC2 t3.small: ~$15/month
- Data transfer: ~$1/month
- Total: ~$16/month

## Support

For issues or questions:
- Check logs: `logs/live/`
- Review health check output
- Consult CLAUDE.md for development commands
- Check Alpaca API status: https://status.alpaca.markets/
