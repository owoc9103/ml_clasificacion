# Cloud Deployment Files (legacy)

> **Curso académico:** use GitHub Actions y Python local — ver [`docs/ACADEMIC_SETUP.md`](../docs/ACADEMIC_SETUP.md). Docker y AWS no son obligatorios.

This directory contains everything needed to deploy your quantitative trading system to a cloud VM.

## 📁 Files

- **`QUICKSTART.md`** - Deploy to AWS in 10 minutes (start here!)
- **`DEPLOYMENT.md`** - Complete deployment guide with all options
- **`aws_ec2_setup.sh`** - Automated setup script for AWS EC2
- **`setup_cron.sh`** - Configure automated daily trading

## 🚀 Quick Deploy

**The fastest way to get started:**

1. **Launch AWS EC2 instance** (Ubuntu 22.04, t3.micro)
2. **SSH into the instance**
3. **Run setup script:**
   ```bash
   curl -O https://raw.githubusercontent.com/your-repo/deploy/aws_ec2_setup.sh
   chmod +x aws_ec2_setup.sh
   ./aws_ec2_setup.sh
   ```
4. **Deploy code and configure** (see QUICKSTART.md)

## 📚 Documentation

### For First-Time Users
→ **Start with `QUICKSTART.md`**
- Step-by-step with copy/paste commands
- Deploy in ~10 minutes
- No Docker/Linux experience needed

### For Production Deployment
→ **Read `DEPLOYMENT.md`**
- Detailed explanations
- Monitoring & maintenance
- Security best practices
- Troubleshooting guide

## 🏗️ Architecture

```
Cloud VM (AWS EC2 / GCP / Azure)
├── Docker Container
│   ├── Python 3.11
│   ├── Trading System Code
│   ├── Trained Models
│   └── Dependencies
├── Cron Job
│   └── Runs Mon-Fri at 9:50 AM ET
├── Logs
│   └── Daily trade logs
└── Monitoring
    └── Health checks
```

## 💰 Estimated Costs

| Provider | Instance Type | Cost/Month |
|----------|--------------|------------|
| AWS EC2 | t3.micro (free tier) | $0 (first 12 months) |
| AWS EC2 | t3.micro | $7/month after free tier |
| AWS EC2 | t3.small (recommended) | $15/month |
| Google Cloud | e2-micro | $6/month |
| Azure | B1s | $10/month |

## 🔐 Security Notes

- Never commit `.env` file (contains API keys)
- Restrict SSH access to your IP only
- Use SSH keys, not passwords
- Rotate API keys periodically
- Monitor logs for suspicious activity

## 🧪 Testing

Before deploying:
```bash
# Test locally with Docker
docker-compose build
docker-compose up

# Run health check
python scripts/health_check.py

# Test trading script
python scripts/run_live.py --mode paper
```

## 📊 Monitoring

After deployment:
```bash
# Check if running
docker-compose ps

# View logs
tail -f logs/live/trading_*.log

# Health check
python scripts/health_check.py

# Check cron schedule
crontab -l
```

## 🆘 Troubleshooting

**Common Issues:**

1. **"Permission denied" on SSH**
   ```bash
   chmod 400 your-key.pem
   ```

2. **Container won't start**
   ```bash
   docker-compose logs
   docker-compose build --no-cache
   ```

3. **API errors**
   - Check `.env` file has correct keys
   - Verify Alpaca account is active

4. **Cron not running**
   ```bash
   sudo systemctl status cron
   ./scripts/run_scheduled.sh  # Test manually
   ```

For more help, see `DEPLOYMENT.md` troubleshooting section.

## 🔄 Updates & Maintenance

```bash
# Pull latest code
git pull

# Rebuild container
docker-compose build
docker-compose up -d

# Check system health
python scripts/health_check.py

# Clean up old logs (keep last 30 days)
find logs/live -name "*.log" -mtime +30 -delete
```

## 📈 Next Steps After Deployment

1. ✅ Monitor for 1-2 weeks in paper mode
2. ✅ Review performance vs backtest
3. ✅ Set up Slack/email notifications
4. ✅ Optimize risk parameters based on live results
5. ✅ Consider Phase 2: Volatility options strategy

---

**Need Help?** Check the full guides or open an issue on GitHub.
