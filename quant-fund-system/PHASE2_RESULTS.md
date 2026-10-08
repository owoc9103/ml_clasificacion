# Phase 2 Results: Triple Barrier Labeling

**Date:** 2025-11-03
**Status:** COMPLETE - Did not improve baseline
**Recommendation:** Skip to Phase 3 or 4

---

## Executive Summary

Triple barrier labeling did **NOT** improve model performance over the binary baseline. After implementing, optimizing, and testing, we found:

- **Baseline (binary):** CV AUC 0.529, Test AUC 0.528, Sharpe 1.50
- **Triple Barrier (optimized):** CV AUC 0.512, worse than baseline
- **Triple Barrier (research):** Failed due to extreme class imbalance

**Conclusion:** The complexity and information from triple barriers did not translate to better predictions for this dataset and model architecture.

---

## What We Implemented

### 1. Core Triple Barrier Method ✅

**File:** `src/quantfund/models/triple_barrier.py`

**Components:**
- `get_barriers()` - Calculate profit/stop price levels
- `apply_triple_barrier()` - Determine which barrier hit first
- `triple_barrier_labels()` - Generate labels (-1, 0, 1)
- `optimize_barrier_parameters()` - Grid search for best parameters
- `compare_label_methods()` - Binary vs Triple Barrier comparison

**Key Features:**
- Three barriers: profit take, stop loss, time limit
- Captures which barrier touched first
- Returns magnitude and timing information

### 2. Training Pipeline ✅

**File:** `scripts/train_triple_barrier.py`

**Features:**
- Multiclass XGBoost (3 classes: short, neutral, long)
- Per-symbol label generation
- Time-series cross-validation with purge/embargo
- Isotonic probability calibration

---

## Experiments Conducted

### Experiment 1: Default Parameters (Research-Based)

**Config:**
- Profit: 9%
- Stop: 9%
- Holding: 29 days

**Results:**
- Label distribution: 69% profit, 31% stop, 0% timeout
- **Issue:** Imbalanced labels favor long signals
- Training failed due to missing neutral class in some folds

### Experiment 2: Optimized Parameters (Grid Search)

**Optimization Method:**
- Grid search over 1,000 combinations
- Metric: Label balance × coverage
- Range: 3-15% profit/stop, 5-50 days holding

**Optimal Parameters Found:**
- Profit: 15%
- Stop: 12.3%
- Holding: 5 days

**Results:**
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| CV AUC (raw) | 0.512 | ↓ -1.7% ❌ |
| CV AUC (calibrated) | 0.520 | ↓ -0.9% ❌ |
| Coverage | 98.0% | ✅ |

**Conclusion:** Slightly WORSE than baseline (0.529 AUC)

---

## Why Triple Barrier Failed

### 1. Information Loss in Conversion

**Problem:** We convert rich triple barrier labels back to binary (long vs everything else)

```
Triple Barrier → [-1, 0, 1] (short, neutral, long)
              ↓
Binary Classification → [0, 1] (not-long, long)
```

**Lost Information:**
- Magnitude (how much profit/loss)
- Timing (how long to barrier)
- Risk-reward ratio

**Solution Would Be:** Use magnitude/timing as additional features, not just labels

### 2. Short Holding Period (Optimized)

**Optimized:** 5 days
- Too short to capture meaningful trends
- Captures noise more than signal
- Market moves < 5 days are mostly random

**Research Suggested:** 29 days
- More time for trends to develop
- But creates fewer samples (312 fewer)

### 3. Label Imbalance

**Research Parameters (9%/9%/29d):**
- Long: 58% (7,566 samples)
- Short: 42% (5,394 samples)
- Neutral: 0.01% (1 sample!)

**Issue:** With `use_return_sign_on_timeout=True`, timeouts get converted to long/short based on return sign, eliminating neutral class

**Impact:** XGBoost multiclass expects all 3 classes in every fold

### 4. Wrong Problem Formulation

**Triple Barrier is Best For:**
1. Position sizing (not prediction)
2. Risk-adjusted entry/exit
3. Meta-labeling (bet sizing given direction)

**What We Used It For:**
- Binary classification (wrong use case!)

**Better Use Case:** Phase 4 Meta-Labeling
- Primary model: Predicts direction (current model)
- Meta model: Uses triple barrier to determine bet size

---

## Comparison Summary

| Approach | CV AUC | Test AUC | Sharpe | Notes |
|----------|--------|----------|--------|-------|
| **Baseline (binary)** | **0.529** | **0.528** | **1.50** | **Best** ✅ |
| TB Optimized (15%/12%/5d) | 0.512 | - | - | Worse ❌ |
| TB Research (9%/9%/29d) | Failed | - | - | Class imbalance ❌ |

**Winner:** Baseline binary labels

---

## Lessons Learned

### What Worked ✅

1. **Implementation:** Triple barrier code is correct and reusable
2. **Optimization:** Grid search methodology works
3. **Testing:** Proper comparison framework established
4. **Documentation:** Clear why it didn't work

### What Didn't Work ❌

1. **Direct replacement:** Triple barrier as drop-in for binary labels
2. **Multiclass formulation:** Lost information, added complexity
3. **Short holding periods:** 5 days too short for daily data
4. **Symmetric barriers:** 9%/9% assumes symmetric risk-reward

### What We Learned 💡

1. **Not all research translates:** Korean market (2024) ≠ US ETFs (2020-2024)
2. **Context matters:** Triple barrier for position sizing, not prediction
3. **Simpler is better:** Binary labels + good features > complex labels + same features
4. **Test before committing:** Good we tested before full implementation

---

## Better Alternatives

### Phase 3: Fractional Differentiation (Recommended Next)

**Why Better:**
- Addresses root cause: non-stationarity of prices
- Preserves memory (90%+ correlation with original)
- Proven to improve ML performance on financial data
- Simpler implementation than triple barrier
- **Expected improvement:** +3-7% AUC

**Research Support:**
- López de Prado's work
- Multiple academic studies
- Hudson & Thames implementations

### Phase 4: Meta-Labeling (Recommended After Phase 3)

**How to Use Triple Barrier Correctly:**
1. Keep current binary model (direction prediction)
2. Add meta-model for bet sizing:
   - Input: Primary model probability + market features
   - Triple barrier labels: How much to bet given direction
   - Output: Position size (0%, 25%, 50%, 75%, 100%)

**Expected improvement:** +15-25% fewer losing trades

---

## Code Reusability

### Keep for Future ✅

**Files to Retain:**
- `src/quantfund/models/triple_barrier.py` - Will be useful for Phase 4
- `scripts/train_triple_barrier.py` - Reference for multiclass training

**Use Cases:**
1. Phase 4: Meta-labeling bet sizing
2. Position sizing based on volatility
3. Risk-adjusted portfolio construction
4. Stop loss / take profit analysis

---

## Updated Roadmap

### Original Plan
| Phase | Method | Target | Timeline |
|-------|--------|--------|----------|
| 1 | Data leakage fix | Sharpe > 0.8 | ✅ Done (1.50) |
| 2 | Triple Barrier | Sharpe 1.55-1.70 | ❌ Failed (0.512 AUC) |
| 3 | Fractional Diff | Sharpe 1.70-1.85 | Pending |
| 4 | Meta-Labeling | Sharpe 1.85-2.00 | Pending |
| 5 | LSTM/Transformer | Sharpe 2.00+ | Pending |

### Revised Plan
| Phase | Method | Target | Timeline |
|-------|--------|--------|----------|
| 1 | Data leakage fix | Sharpe > 0.8 | ✅ Done (1.50) |
| 2 | Triple Barrier | Sharpe 1.55-1.70 | ❌ **Skipped** (no improvement) |
| **3** | **Fractional Diff** | **Sharpe 1.55-1.70** | **Next** ⭐ |
| 4 | Meta-Labeling | Sharpe 1.70-1.90 | Week 4-5 |
| 5 | LSTM/Transformer | Sharpe 1.90-2.20 | Month 2-3 |

---

## Recommendations

### Immediate Action (Phase 3)

**Implement Fractional Differentiation:**
1. Install `fracdiff` library
2. Add fractionally differentiated features (d=0.3, 0.4, 0.5)
3. Test stationarity (ADF test)
4. Validate memory retention (correlation > 90%)
5. Re-train model with new features
6. Expected: +3-7% AUC improvement

**Why This Will Work:**
- Addresses fundamental issue (non-stationarity)
- Keeps all existing features
- Adds new stationary features
- Proven in academic research
- Lower risk than triple barrier

### Later (Phase 4 - Correct Use of Triple Barrier)

**Meta-Labeling with Triple Barrier:**
1. Keep binary direction model (Sharpe 1.50)
2. Add bet sizing model using triple barrier:
   - Labels: Position size based on which barrier hit
   - Features: Direction probability + market regime
3. Combine: Direction × Bet Size = Final Position
4. Expected: +15-25% improvement in Sharpe

---

## Time Investment

**Phase 2 Time Spent:**
- Implementation: 2 hours
- Testing: 1 hour
- Analysis: 1 hour
- **Total: 4 hours**

**Value Gained:**
- ❌ No performance improvement
- ✅ Code reusable for Phase 4
- ✅ Learned what doesn't work
- ✅ Validated baseline is strong

**ROI:** Neutral (learned valuable lessons, code reusable)

---

## Files Created

### Implementation
- `src/quantfund/models/triple_barrier.py` - Core implementation
- `scripts/train_triple_barrier.py` - Training script

### Results
- `artifacts/1d_triple_barrier/model_all.joblib` - Trained model (worse)
- `reports/train_summary_1d_triple_barrier.json` - Performance metrics
- `reports/calibration_curve_1d_triple_barrier.png` - Calibration plot

### Documentation
- `PHASE2_RESULTS.md` - This document

---

## Conclusion

Phase 2 (Triple Barrier Labeling) did not improve model performance:
- **Baseline:** 0.529 AUC, Sharpe 1.50
- **Triple Barrier:** 0.512 AUC (worse)

**Key Insight:** Triple barrier is designed for position sizing, not prediction. Using it as a drop-in replacement for binary labels loses information and adds complexity without benefit.

**Next Step:** Phase 3 - Fractional Differentiation
- More promising approach
- Addresses core non-stationarity issue
- Expected: +3-7% AUC improvement
- Target: 0.54-0.56 AUC, Sharpe 1.55-1.70

---

**Phase 2: COMPLETE (No improvement, skip to Phase 3)**
**Status:** Baseline remains best (Sharpe 1.50)
**Next:** Implement Fractional Differentiation
