# Cloud VM Deployment Guide

Complete guide to deploying your quantitative trading system to a cloud VM.

## Overview

This deployment uses:
- **Docker** - Containerized application for consistency
- **Cron** - Automated scheduling (runs Mon-Fri at 9:50 AM ET)
- **Logging** - All trades logged with timestamps
- **Monitoring** - Optional email/Slack alerts on errors

## Cloud Provider Options

### AWS EC2 (Recommended)
- **Cost:** ~$5-10/month (t3.micro or t3.small)
- **Region:** us-east-1 (N. Virginia) for low latency
- **OS:** Ubuntu 22.04 LTS
- **Storage:** 20GB

### Google Cloud Compute Engine
- **Cost:** ~$5-8/month (e2-micro or e2-small)
- **Region:** us-east1 or us-central1
- **OS:** Ubuntu 22.04 LTS

### Azure Virtual Machine
- **Cost:** ~$8-12/month (B1s or B2s)
- **Region:** East US
- **OS:** Ubuntu 22.04 LTS

## Step-by-Step Deployment

### 1. Create Cloud VM

#### AWS EC2:
```bash
# Via AWS Console:
1. Go to EC2 Dashboard
2. Launch Instance
3. Choose Ubuntu 22.04 LTS
4. Instance type: t3.micro (free tier eligible) or t3.small
5. Configure security group:
   - Allow SSH (port 22) from your IP
6. Create/select key pair for SSH access
7. Launch instance
```

#### Manual CLI (AWS):
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 LTS
  --instance-type t3.micro \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=quantfund-trader}]'
```

### 2. Connect to VM

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<your-vm-ip>
```

### 3. Run Setup Script

```bash
# Download and run the setup script
curl -O https://raw.githubusercontent.com/your-repo/deploy/aws_ec2_setup.sh
chmod +x aws_ec2_setup.sh
./aws_ec2_setup.sh

# Log out and back in for docker group to take effect
exit
ssh -i your-key.pem ubuntu@<your-vm-ip>
```

### 4. Deploy Your Code

#### Option A: Git Clone (Recommended)
```bash
cd ~/quantfund
git clone https://github.com/your-username/quant-fund-mvp.git .
```

#### Option B: SCP Upload
```bash
# From your local machine
scp -i your-key.pem -r C:\projects\quant-fund-mvp/* ubuntu@<vm-ip>:~/quantfund/
```

### 5. Configure Environment Variables

```bash
cd ~/quantfund

# Create .env file with your API keys
nano .env
```

Add:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

Save and exit (Ctrl+O, Enter, Ctrl+X)

### 6. Build and Start Container

```bash
# Build the Docker image
docker-compose build

# Start the container
docker-compose up -d

# Check status
docker-compose ps
```

### 7. Set Up Automated Scheduling

```bash
# Make scripts executable
chmod +x scripts/run_scheduled.sh
chmod +x deploy/setup_cron.sh

# Set up cron job (runs Mon-Fri at 9:50 AM ET)
./deploy/setup_cron.sh
```

### 8. Test the System

```bash
# Manual test run
docker-compose exec trading-bot python scripts/run_live.py --mode paper

# Or test the scheduled script directly
./scripts/run_scheduled.sh

# Check logs
tail -f logs/live/trading_*.log
```

## Monitoring & Maintenance

### View Logs

```bash
# Real-time log viewing
tail -f logs/live/trading_$(date +%Y%m%d)*.log

# View all logs from today
cat logs/live/trading_$(date +%Y%m%d)*.log

# Search for errors
grep -i error logs/live/*.log
```

### Check Cron Status

```bash
# View cron schedule
crontab -l

# View cron execution logs
grep CRON /var/log/syslog
```

### Container Management

```bash
# View running containers
docker-compose ps

# View container logs
docker-compose logs -f trading-bot

# Restart container
docker-compose restart

# Stop container
docker-compose stop

# Start container
docker-compose start

# Rebuild after code changes
git pull
docker-compose build
docker-compose up -d
```

### Resource Monitoring

```bash
# Check CPU/Memory usage
htop

# Check disk usage
df -h

# Check Docker resource usage
docker stats
```

## Email/Slack Notifications (Optional)

### Email Setup (via SendGrid/SES)

Edit `scripts/run_scheduled.sh` and uncomment the email section:
```bash
# Install mailutils
sudo apt-get install -y mailutils

# Configure in run_scheduled.sh:
echo "Trading run failed. See ${LOG_FILE}" | mail -s "Trading Error" your-email@example.com
```

### Slack Webhook Setup

1. Create a Slack webhook: https://api.slack.com/messaging/webhooks
2. Edit `scripts/run_scheduled.sh` and add:
```bash
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"Trading run failed at $(date)\"}" \
  "${SLACK_WEBHOOK}"
```

## Security Best Practices

1. **Firewall Rules:**
   - Only allow SSH from your IP
   - No need to expose any other ports

2. **Key Management:**
   - Keep `.env` file secure (never commit to git)
   - Use SSH keys, not passwords
   - Rotate API keys periodically

3. **Updates:**
   ```bash
   # Update system packages monthly
   sudo apt-get update && sudo apt-get upgrade -y

   # Update Docker images
   docker-compose pull
   docker-compose up -d
   ```

4. **Backups:**
   ```bash
   # Backup logs and data
   tar -czf backup_$(date +%Y%m%d).tar.gz logs/ data/ .env

   # Upload to S3 (optional)
   aws s3 cp backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
   ```

## Cost Optimization

### AWS Free Tier (First 12 months)
- t3.micro: 750 hours/month free
- After free tier: ~$7/month

### Reserved Instances (1-year commit)
- Save ~30-40% vs on-demand
- Good if you plan to run long-term

### Auto-Stop During Non-Trading Hours
```bash
# Add to crontab to stop VM at 5 PM ET (21:00 UTC)
0 21 * * 1-5 sudo shutdown -h now

# Set up AWS Lambda to start it at 9:30 AM ET
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Cron Not Running
```bash
# Check cron service
sudo systemctl status cron

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Test script manually
./scripts/run_scheduled.sh
```

### API Authentication Errors
```bash
# Verify .env file exists and has correct keys
cat .env

# Restart container to reload environment
docker-compose restart
```

### Out of Disk Space
```bash
# Clean up old Docker images
docker system prune -a

# Clean up old logs (older than 30 days)
find logs/live -name "*.log" -mtime +30 -delete
```

## Switching from Paper to Live Trading

⚠️ **WARNING: Live trading uses real money!**

1. Update `.env` to use live API keys
2. Change config: `configs/live_trading.yaml`
   ```yaml
   broker:
     base_url: "https://api.alpaca.markets"  # Remove 'paper-'
   ```
3. Test thoroughly in paper mode first
4. Start with small capital
5. Monitor closely for first week

## Support

- Alpaca API Docs: https://alpaca.markets/docs
- Docker Docs: https://docs.docker.com
- Ubuntu Server Guide: https://ubuntu.com/server/docs

## Next Steps

After deployment:
1. ✅ Monitor for 1-2 weeks in paper mode
2. ✅ Review trades daily via Alpaca dashboard
3. ✅ Track performance vs backtest
4. ✅ Adjust risk parameters if needed
5. ✅ Consider Phase 2: Volatility options strategy
