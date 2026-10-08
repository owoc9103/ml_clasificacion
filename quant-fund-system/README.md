# 🤖 Autonomous ML-Powered Trading System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-deployed-orange.svg)](https://aws.amazon.com/)

A production-grade quantitative trading system that combines machine learning with technical analysis to trade ETFs autonomously. Built with XGBoost, deployed on AWS, and integrated with Alpaca Markets for commission-free trading.

### Versión académica (ML Fin Econ)

- **Sin Docker ni AWS obligatorios:** Python local + [GitHub Actions](.github/workflows/academic-pipeline.yml). Guía: [docs/ACADEMIC_SETUP.md](docs/ACADEMIC_SETUP.md).
- **Varios modelos sklearn + XGBoost** (SVM, árboles, Random Forest, etc.) con comparación en `reports/model_comparison_*.json`.
- **Universo:** `configs/universe.yaml` (13 tickers originales).

## Medium Article
- [Medium](https://medium.com/@erenari27/building-an-autonomous-quant-system-from-backtest-to-aws-deployment-ec6121cb6dfe)

## 🎯 Overview

This system automatically:
- Analyzes **12 ETFs** daily using 20+ technical indicators
- Generates trading signals via **calibrated ML models** (XGBoost + isotonic regression)
- Executes trades through **Alpaca API** with sophisticated risk management
- Runs **autonomously on AWS EC2** (Mon-Fri at 9:50 AM ET)
- Logs everything for monitoring and performance analysis

**Current Status:** ✅ Deployed and operational in paper trading mode

---

## 📊 Performance

### Backtest Results (2020-01-01 to 2025-09-27)

| Metric | Value |
|--------|-------|
| **Annual Return** | 1.73% |
| **Annual Volatility** | 3.58% |
| **Sharpe Ratio** | 0.48 |
| **Max Drawdown** | -10.05% |
| **Total Trades** | 35,286 |
| **Avg Trades/Year** | ~7,057 |
| **Strategy Mix** | 30% Momentum / 70% Mean Reversion |
| **Trading Frequency** | Daily rebalancing |

**Key Characteristics:**
- ✅ **Low volatility:** 3.6% annual vol vs SPY's ~18% (defensive strategy)
- ✅ **Risk-controlled:** Drawdown stayed under 10.1% (below 12% limit)
- ✅ **Positive Sharpe:** 0.48 risk-adjusted returns
- ⚠️ **High frequency:** 35k+ trades over 5 years (transaction costs matter)
- 💡 **Ideal for:** Capital preservation with modest returns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  yfinance → Parquet Storage → Feature Engineering (20+ ind) │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  ML Model Layer                              │
├─────────────────────────────────────────────────────────────┤
│  XGBoost Classifier → Isotonic Calibration → Probabilities  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Signal Generation                            │
├─────────────────────────────────────────────────────────────┤
│  Momentum Signals + Mean-Reversion Signals → Combined       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Risk Management & Sizing                        │
├─────────────────────────────────────────────────────────────┤
│  Kelly Criterion → Volatility Targeting → Position Limits   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Order Execution                               │
├─────────────────────────────────────────────────────────────┤
│  Alpaca API → Market Orders → Confirmation → Logging        │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Alpaca account (free at [alpaca.markets](https://alpaca.markets))
- AWS account (optional, for deployment)

### Local Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/ErenAri/quant-fund-system.git
cd quant-fund-system

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure API keys
cp .env.example .env
nano .env  # Add your Alpaca API keys

# 5. Test connection
python scripts/health_check.py

# 6. Run paper trading
python scripts/run_live.py --mode paper
```

### AWS Deployment (10 minutes)

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for complete AWS EC2 setup instructions.

**Quick version:**

```bash
# On EC2 instance:
git clone https://github.com/ErenAri/quant-fund-system.git
cd quant-fund-system
cp .env.example .env && nano .env  # Add API keys
docker-compose build && docker-compose up -d
./deploy/setup_cron.sh  # Automated daily trading
```

---

## 🎓 Key Features

### Machine Learning
- **XGBoost binary classifier** with cost-aware labels
- **Time-series cross-validation** with purge & embargo (prevents overfitting)
- **Isotonic calibration** for accurate probability estimates
- **Walk-forward retraining** quarterly to adapt to market changes
- **Advanced labeling methods:** Triple barrier labeling, fractional differentiation, meta-labeling for bet sizing

### Signal Generation
- **Momentum signals:** ML probabilities + trend indicators (MACD, ADX, EMA crossovers)
- **Mean-reversion signals:** RSI oversold + Bollinger Bands + z-scores
- **Regime filtering:** Market trend + volatility + credit spreads (optional)

### Risk Management
- **Kelly criterion** position sizing capped at 30%
- **Volatility targeting:** 10-20% annual portfolio volatility
- **Per-trade risk limit:** 1.5% maximum
- **Daily loss stop:** 1% circuit breaker
- **ATR-based stops:** 3x multiplier trailing stops
- **Position concentration:** Maximum 30% in any single asset

### Trading Universe
**12 Liquid ETFs** across asset classes:
- **Equity Indices:** SPY, QQQ, IWM, DIA
- **Sector Rotation:** XLK (Tech), XLF (Financials), XLE (Energy), XLY (Consumer Discretionary), XLP (Consumer Staples), XLV (Healthcare)
- **Fixed Income:** TLT (20Y Treasury)
- **Credit:** HYG (High Yield Bonds)

### Infrastructure
- **Docker containerization** for reproducible deployments
- **Cron scheduling** for automated daily execution
- **Comprehensive logging** with structured JSON output
- **Health monitoring** with system checks
- **AWS-ready** with EC2 deployment scripts

---

## 📁 Project Structure

```
quant-fund-system/
├── src/quantfund/              # Core trading library
│   ├── data/                   # Data ingestion & validation
│   │   ├── ingest.py          # yfinance wrapper
│   │   └── validators.py       # Data quality checks
│   ├── features/               # Feature engineering
│   │   ├── core.py            # Technical indicators (49+)
│   │   └── fracdiff.py        # Fractional differentiation (FFD)
│   ├── models/                 # ML model training
│   │   ├── train.py           # XGBoost + calibration
│   │   ├── triple_barrier.py  # Triple barrier labeling
│   │   └── meta_labeling.py   # Meta-labeling for bet sizing
│   ├── strategies/             # Signal generation
│   │   ├── momo.py            # Momentum signals
│   │   ├── meanrev.py         # Mean-reversion signals
│   │   └── filters.py         # Regime detection
│   ├── risk/                   # Risk management
│   │   └── controls.py        # Position sizing, stops
│   ├── backtest/               # Backtesting engine
│   │   └── engine.py          # Event-driven backtest
│   ├── execution/              # Live trading
│   │   ├── alpaca_broker.py   # Alpaca API client
│   │   └── live_engine.py     # Production engine
│   └── utils/                  # Utilities
│       └── log.py             # Structured logging
├── scripts/                    # Executable scripts
│   ├── fetch_data.py          # Download OHLCV data
│   ├── make_dataset.py        # Compute features
│   ├── train_model.py         # Train ML models
│   ├── train_triple_barrier.py # Train with triple barrier labels
│   ├── train_meta_model.py    # Train meta-labeling model
│   ├── evaluate_meta_model.py # Evaluate meta-model performance
│   ├── test_fracdiff.py       # Test fractional differentiation
│   ├── run_backtest.py        # Run backtest
│   ├── run_live.py            # Live/paper trading
│   ├── run_scheduled.sh       # Cron wrapper
│   └── health_check.py        # System monitoring
├── configs/                    # Configuration files
│   ├── universe.yaml          # Trading symbols
│   ├── train.yaml             # ML training params
│   ├── data.yaml              # Data fetch params
│   └── live_trading.yaml      # Live trading config
├── deploy/                     # Deployment scripts
│   ├── DEPLOYMENT.md          # Full deployment guide
│   ├── QUICKSTART.md          # Quick setup
│   ├── aws_ec2_setup.sh       # EC2 provisioning
│   └── setup_cron.sh          # Cron configuration
├── tests/                      # Unit & property tests
├── data/                       # Data storage
│   ├── parquet/               # Raw OHLCV (partitioned)
│   └── datasets/              # Feature datasets
├── artifacts/                  # Trained models
│   ├── 1d/                    # Daily models
│   ├── 60m/                   # Hourly models
│   └── 120m/                  # 2-hour models
├── logs/                       # Application logs
│   └── live/                  # Trading execution logs
├── reports/                    # Backtest reports
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker orchestration
├── pyproject.toml             # Python package config
├── requirements.txt           # Dependencies
├── .env.example               # API key template
├── CLAUDE.md                  # Development guide
├── DEPLOYMENT_GUIDE.md        # AWS deployment
└── README.md                  # This file
```

---

## 🔧 Configuration

All settings in `configs/live_trading.yaml`:

```yaml
# Strategy parameters
strategy:
  interval: "1d"                # Daily rebalancing
  momo_weight: 0.3              # 30% momentum, 70% mean-reversion
  use_regime_filter: false      # Optional market filter
  use_vol_regime: false         # Optional volatility filter

# Risk controls
risk:
  annual_vol_target: 0.20       # 20% target volatility
  max_drawdown: 0.12            # 12% max drawdown cutoff
  per_trade_risk: 0.015         # 1.5% per-trade risk limit
  daily_loss_stop: 0.01         # 1% daily loss circuit breaker
  atr_stop_mult: 3.0            # ATR stop multiplier
  kelly_cap: 0.30               # Max 30% Kelly fraction

# Position limits
limits:
  max_position_size: 0.30       # Max 30% in single position
  min_position_size: 0.01       # Min 1% position
  max_num_positions: 12         # Max concurrent positions

# Transaction costs
costs:
  commission_bps: 0.0           # Alpaca is commission-free
  slippage_bps: 1.0             # Assume 1bp slippage

# Execution
execution:
  order_type: "market"          # Market orders
  time_in_force: "day"          # Day orders
  fill_timeout_seconds: 30      # Order timeout

# Schedule
schedule:
  trading_days: ["MON", "TUE", "WED", "THU", "FRI"]
  signal_generation_time: "09:45"  # 15min after open
  execution_time: "09:50"          # Execute at 9:50 AM ET
```

---

## 🧪 Development Workflow

### Data Pipeline

```bash
# 1. Fetch raw OHLCV data
python scripts/fetch_data.py --start 2012-01-01 --end 2025-09-27

# 2. Compute features (technical indicators)
python scripts/make_dataset.py

# 3. Train ML model with time-series CV
python scripts/train_model.py --start 2012-01-01 --end 2025-09-27 --interval 1d

# 4. Run backtest
python scripts/run_backtest.py --interval 1d --use_model

# 5. Deploy to paper trading
python scripts/run_live.py --mode paper
```

### Testing

```bash
# Run all tests with coverage (80% minimum)
python -m pytest tests/ -v --cov=src/quantfund --cov-report=html

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run specific test
python -m pytest tests/test_strategies.py::TestMomentumStrategy::test_momo_signals_basic_functionality -v

# Property-based testing with hypothesis
python -m pytest tests/test_strategies.py::TestMomentumStrategy::test_momo_signals_property_based -v
```

---

## 📊 Monitoring

### View Logs

```bash
# On EC2 instance
tail -f ~/quantfund/logs/live/trading_*.log

# View latest log
ls -lt logs/live/ | head -5
cat logs/live/trading_20251003_095000.log
```

### Health Check

```bash
docker-compose exec trading-bot python scripts/health_check.py
```

Expected output:
```
[PASS] Credentials - OK
[PASS] Broker Connection - Connected (Cash: $100,234.39)
[PASS] Model Files - OK (age: 7 days)
[PASS] Recent Activity - Last run: 2h ago
[PASS] Disk Space - OK: 45GB free
```

### Alpaca Dashboard

Monitor positions, orders, and P&L:
- **Paper:** https://app.alpaca.markets/paper/dashboard/overview
- **Live:** https://app.alpaca.markets/live/dashboard/overview

---

## 💰 Cost Breakdown

### Infrastructure

| Service | Instance | Monthly Cost |
|---------|----------|--------------|
| AWS EC2 | t3.small (2 vCPU, 2GB) | $15 |
| Data transfer | < 1GB/month | $1 |
| **Total** | | **~$16/month** |

> 💡 **Free Tier:** Use AWS t3.micro for free (first 12 months)

### Trading Costs

- **Alpaca commission:** $0 (commission-free)
- **Slippage:** ~1 bp assumed (market-dependent)
- **SEC fees:** ~$0.0008 per $100 (negligible)

---

## 🔐 Security Best Practices

✅ **API keys in `.env`** - Never commit credentials
✅ **`.gitignore` configured** - Sensitive files excluded
✅ **SSH key authentication** - No password login
✅ **Firewall rules** - Only allow SSH from your IP
✅ **Regular key rotation** - Change API keys every 90 days
✅ **Paper trading first** - Test thoroughly before live

---

## 🗺️ Roadmap

### ✅ Phase 1: Core System (Complete)
- [x] ETF momentum + mean-reversion strategy
- [x] XGBoost ML model with calibration
- [x] Risk management framework
- [x] Alpaca broker integration
- [x] Docker containerization
- [x] AWS EC2 deployment
- [x] Automated cron scheduling
- [x] Comprehensive logging

### 🚧 Phase 2: Enhancements (In Progress)
- [ ] Slack/email notifications
- [ ] Real-time monitoring dashboard
- [ ] Additional regime filters (VIX, credit spreads)
- [ ] Multi-interval strategies (1h, 4h)
- [ ] Performance analytics reports

### 🔮 Phase 3: Advanced Features (Future)
- [ ] Alternative data integration (sentiment, positioning)
- [ ] Reinforcement learning for adaptive strategies
- [ ] Options strategies (iron condors, spreads)
- [ ] Multi-asset expansion (forex, crypto, commodities)
- [ ] Portfolio optimization (mean-variance, risk parity)
- [ ] Custom execution algorithms (TWAP, VWAP)

---

## 📚 Resources & References

### Books
- **Advances in Financial Machine Learning** - Marcos López de Prado
- **Quantitative Trading** - Ernest Chan
- **Machine Learning for Asset Managers** - Marcos López de Prado

### Documentation
- [Alpaca API Docs](https://alpaca.markets/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)

### Related Projects
- [QuantConnect](https://www.quantconnect.com/) - Algorithmic trading platform
- [Backtrader](https://www.backtrader.com/) - Python backtesting library
- [Zipline](https://github.com/quantopian/zipline) - Quantopian's backtester

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:**
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Maintain 80%+ code coverage

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- ⚠️ Trading involves substantial risk of loss
- ⚠️ Past performance does not guarantee future results
- ⚠️ No warranty or guarantee of profitability
- ⚠️ Use at your own risk
- ⚠️ Not financial advice
- ⚠️ Author is not a registered investment advisor
- ⚠️ Always test thoroughly with paper trading before risking real capital

By using this software, you acknowledge that:
- You understand the risks of algorithmic trading
- You are responsible for your own trading decisions
- You will comply with all applicable laws and regulations
- You will not hold the author liable for any losses

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Alpaca Markets** for providing free commission trading API
- **XGBoost team** for the excellent ML library
- **Marcos López de Prado** for pioneering work in ML for finance
- **Open source community** for the amazing Python ecosystem

---

## 📬 Contact

- **GitHub:** [@ErenAri](https://github.com/ErenAri)
- **Issues:** [GitHub Issues](https://github.com/ErenAri/quant-fund-system/issues)

---

**⭐ If you find this project useful, please consider giving it a star!**

---

*Last Updated: October 2025*
*Status: ✅ Operational | Mode: Paper Trading | Instance: AWS EC2 t3.small*
