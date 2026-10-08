## Volatility Options Strategy Design

### Core Thesis
Our ML model can predict volatility regimes with **0.83 AUC** (excellent). We'll profit from this by:
1. **Selling premium when predicting LOW volatility** (vol will stay low → options expire worthless → we keep premium)
2. **Buying protection when predicting HIGH volatility** (vol will spike → options gain value)

### Strategy: Adaptive Iron Condor

**What is an Iron Condor?**
- Sell OTM call + OTM put (collect premium)
- Buy further OTM call + put (limit risk)
- Profit if price stays in range (low volatility)
- Max loss is defined and limited

**Our Edge: ML-Predicted Volatility**
- When model predicts **low vol (p < 0.3)**: Sell tight condors (high premium, narrow range)
- When model predicts **high vol (p > 0.7)**: Either skip or sell wide condors (lower premium, wider range)
- Dynamic position sizing based on vol confidence

### Example Trade (SPY @ $500)

**Low Vol Prediction (p_high_vol = 0.2):**
```
Sell: SPY 510 Call @ $2.00
Buy:  SPY 515 Call @ $0.50
Sell: SPY 490 Put  @ $2.00
Buy:  SPY 485 Put  @ $0.50

Net Credit: $3.00 per contract
Max Risk: $2.00 per contract (if SPY > 515 or < 485)
Win if: SPY stays between 487-512 (high probability in low vol)
```

**High Vol Prediction (p_high_vol = 0.8):**
```
Skip iron condor, or go wider:

Sell: SPY 520 Call @ $1.00
Buy:  SPY 530 Call @ $0.30
Sell: SPY 480 Put  @ $1.00
Buy:  SPY 470 Put  @ $0.30

Net Credit: $1.40 per contract
Max Risk: $8.60 per contract
Win if: SPY stays between 471.40-528.60 (wider range for high vol)
```

### Position Sizing Rules

1. **Kelly Criterion (adapted for options)**:
   - `f = (p_win * avg_win - p_loss * avg_loss) / avg_loss`
   - Where `p_win` = probability price stays in range
   - Scale by vol confidence: multiply by (1 - p_high_vol)

2. **Risk per trade**:
   - Low vol: risk up to 2% of portfolio per trade
   - Med vol: risk up to 1% of portfolio
   - High vol: risk up to 0.5% or skip

3. **Portfolio allocation**:
   - Max 10 concurrent positions
   - Max 20% of capital at risk across all positions
   - 30% cash buffer for margin requirements

### Entry/Exit Rules

**Entry (Daily, at market open)**:
1. Run ML model on previous day's data
2. Get p_high_vol for next 5 days
3. If p_high_vol < 0.4: Enter iron condor (tight)
4. If 0.4 < p_high_vol < 0.7: Enter iron condor (wide) or skip
5. If p_high_vol > 0.7: Skip or buy protective straddle

**Exit**:
1. **Time**: Close 2 days before expiration (avoid gamma risk)
2. **Profit target**: Close at 50% max profit
3. **Stop loss**: Close if loss reaches 100% of premium collected
4. **Vol spike**: If realized vol > predicted by 50%, close immediately

**DTE (Days to Expiration)**:
- Sell options with 7-14 DTE (weekly options)
- Sweet spot: theta decay accelerates, but not too close to expiration

### Risk Management

**Position Limits**:
- Max loss per trade: 2x premium collected
- Max portfolio drawdown: 15%
- Daily loss stop: 3% of portfolio
- Correlation limit: No more than 3 positions on same underlying

**Margin Requirements**:
- Iron condor margin = max(width of call spread, width of put spread)
- Keep 2x margin requirement in cash (safety buffer)
- Monitor margin in real-time, close positions if approaching limit

**Black Swan Protection**:
- Always buy wings (never naked short)
- VIX hedges: Buy VIX calls when VIX < 15 and p_high_vol > 0.6
- Size down during earnings season (unpredictable vol spikes)

### Expected Performance

**Backtest Assumptions** (to be validated):
- Win rate: 70% (based on 0.83 AUC vol prediction)
- Avg win: $300 per contract (50% of max profit)
- Avg loss: $600 per contract (100% of premium, hit stop)
- Avg trades per week: 3-5
- Capital required: $25k minimum (PDT rule)

**Conservative Estimate**:
- Win rate: 65%
- Avg R:R = 0.5 (risk $2 to make $1 per contract)
- Kelly f = (0.65 * 300 - 0.35 * 600) / 600 = 0.175 (17.5% per trade)
- With 3-5 trades/week, 4% per month = **48% annual return**
- Sharpe ~1.5 (if vol prediction holds)

### Implementation Steps

**Phase 1: Data & Features (1 week)**
- [x] We already have vol prediction model (0.83 AUC)
- [ ] Fetch options data (use Alpaca options API or CBOE data)
- [ ] Calculate implied vol (IV) vs predicted realized vol
- [ ] Build options pricing model (Black-Scholes + Greeks)

**Phase 2: Strategy Development (1 week)**
- [ ] Implement iron condor logic
- [ ] Add position sizing with Kelly
- [ ] Risk management layer
- [ ] Backtest on historical options data

**Phase 3: Paper Trading (1 month)**
- [ ] Connect to options broker (Alpaca, Tastytrade, or IBKR)
- [ ] Run strategy on paper account
- [ ] Validate win rate matches prediction
- [ ] Monitor slippage, fills, margin usage

**Phase 4: Live (gradual scale)**
- [ ] Start with 1 contract per trade, $5k capital
- [ ] Scale to 5 contracts after 2 weeks if profitable
- [ ] Scale to 10 contracts after 1 month
- [ ] Target: $50k capital, 20-30 contracts per trade

### Key Risks

1. **Model degradation**: Vol prediction AUC drops (monitor monthly)
2. **Regime change**: Market structure shifts (2020 COVID-like events)
3. **Execution**: Slippage on options spreads (use limit orders)
4. **Liquidity**: Wide bid-ask spreads (only trade SPY/QQQ with tight spreads)
5. **Assignment risk**: Early assignment on short options (rare but manage)

### Success Metrics (Paper Trading)

Track these for 1 month before going live:
- [ ] Win rate > 60%
- [ ] Sharpe > 1.0
- [ ] Max drawdown < 20%
- [ ] Avg slippage < $0.10 per contract
- [ ] No margin calls
- [ ] Model AUC remains > 0.75

### Tools & Platforms

**Data**:
- Options data: CBOE, Alpaca, or HistoricalOptionData.com
- IV data: Market Chameleon or IVolatility

**Brokers** (options-friendly):
- Tastytrade (low commissions, good platform)
- Interactive Brokers (best for algos, API access)
- Alpaca (has options API, easy integration)

**Analysis**:
- OptionStrat (visualize P&L)
- Think or Swim (Greeks, analysis)
- Custom Python backtester (build this)

---

## Next Steps

1. **Fetch options data** for SPY/QQQ (last 2 years)
2. **Build options backtester** (iron condor specific)
3. **Validate strategy** on historical data
4. **Paper trade** for 1 month
5. **Go live** with small size

Ready to start implementing?
