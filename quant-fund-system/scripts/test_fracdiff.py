#!/usr/bin/env python3
"""
Test fractional differentiation implementation.

Validates:
1. Stationarity improvement (original vs fracdiff)
2. Memory retention (correlation > 90%)
3. Optimal d finding
4. Feature generation
"""
import pandas as pd
import numpy as np
from pathlib import Path

from quantfund.features.fracdiff import (
    frac_diff_ffd,
    test_stationarity,
    compute_memory_retention,
    find_min_d,
    add_fracdiff_features,
)


def main():
    print("="*60)
    print("FRACTIONAL DIFFERENTIATION TEST")
    print("="*60)
    print()

    # Load SPY data
    data_path = Path("data/datasets/interval=1d/symbol=SPY/data.parquet")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows of SPY data")
    print()

    # Test on close prices
    prices = df['close'].copy()
    print(f"Testing on close prices (n={len(prices)})")
    print()

    # 1. Test original series stationarity
    print("-" * 60)
    print("1. ORIGINAL SERIES STATIONARITY")
    print("-" * 60)
    is_stat_orig, pval_orig, results_orig = test_stationarity(prices)
    print(f"ADF Statistic: {results_orig['adf_stat']:.4f}")
    print(f"P-value: {pval_orig:.6f}")
    print(f"Is Stationary: {is_stat_orig}")
    print(f"Critical Values: {results_orig['critical_values']}")
    print()

    # 2. Test fractionally differentiated series
    print("-" * 60)
    print("2. FRACTIONALLY DIFFERENTIATED SERIES")
    print("-" * 60)

    d_tests = [0.3, 0.4, 0.5]
    results_table = []

    for d in d_tests:
        print(f"\nTesting d={d:.1f}:")
        print("-" * 40)

        # Apply fractional differentiation
        frac_prices = frac_diff_ffd(prices, d)

        # Test stationarity
        is_stat, pval, adf_results = test_stationarity(frac_prices)

        # Compute memory retention
        memory = compute_memory_retention(prices, frac_prices)

        # Count NaN values (from window)
        n_nan = frac_prices.isna().sum()
        coverage = (len(frac_prices) - n_nan) / len(frac_prices)

        results_table.append({
            'd': d,
            'stationary': is_stat,
            'p_value': pval,
            'adf_stat': adf_results['adf_stat'],
            'memory': memory,
            'coverage': coverage,
            'n_nan': n_nan,
        })

        print(f"  Stationary: {is_stat}")
        print(f"  P-value: {pval:.6f}")
        print(f"  ADF Stat: {adf_results['adf_stat']:.4f}")
        print(f"  Memory Retention: {memory:.2%}")
        print(f"  Coverage: {coverage:.1%} ({len(frac_prices) - n_nan}/{len(frac_prices)} values)")

    # Summary table
    print()
    print("-" * 60)
    print("SUMMARY TABLE")
    print("-" * 60)
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))
    print()

    # 3. Find optimal d
    print("-" * 60)
    print("3. FIND OPTIMAL d")
    print("-" * 60)
    print("Searching for minimum d that achieves stationarity...")
    print("Target: p-value < 0.05, memory > 90%")
    print()

    optimal_d, search_results = find_min_d(
        prices,
        d_range=(0.0, 1.0),
        step=0.05,
        target_pvalue=0.05,
        min_memory=0.90,
    )

    print(f"Optimal d: {optimal_d:.2f}")
    print()
    print("Search results (first 10):")
    print(search_results['search_results'].head(10).to_string(index=False))
    print()

    # 4. Add features to dataframe
    print("-" * 60)
    print("4. ADD FRACDIFF FEATURES")
    print("-" * 60)
    df_with_features = add_fracdiff_features(df, 'close', [0.3, 0.4, 0.5])
    print(f"Original columns: {list(df.columns[:10])}...")
    print(f"New columns: {[c for c in df_with_features.columns if 'fracdiff' in c]}")
    print()
    print("Sample data:")
    print(df_with_features[['close', 'fracdiff_d30', 'fracdiff_d40', 'fracdiff_d50']].tail(10))
    print()

    # 5. Validation summary
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print()

    # Check if any d achieves stationarity
    stationary_d = df_results[df_results['stationary'] == True]
    if len(stationary_d) > 0:
        best_d = stationary_d.iloc[0]
        print(f"[PASS] Found stationary d={best_d['d']:.1f}")
        print(f"       P-value: {best_d['p_value']:.6f} < 0.05")
        print(f"       Memory: {best_d['memory']:.2%}")
    else:
        print("[WARN] No d achieved full stationarity (p < 0.05)")
        best_d = df_results.loc[df_results['p_value'].idxmin()]
        print(f"       Best d={best_d['d']:.1f} with p-value {best_d['p_value']:.6f}")

    # Check memory retention
    high_memory = df_results[df_results['memory'] >= 0.90]
    if len(high_memory) > 0:
        print(f"[PASS] {len(high_memory)}/{len(df_results)} d values retain >90% memory")
        for _, row in high_memory.iterrows():
            print(f"       d={row['d']:.1f}: {row['memory']:.2%}")
    else:
        print(f"[WARN] No d achieves 90% memory retention")
        best_mem = df_results.loc[df_results['memory'].idxmax()]
        print(f"       Best d={best_mem['d']:.1f} with {best_mem['memory']:.2%}")

    # Check data coverage
    good_coverage = df_results[df_results['coverage'] >= 0.95]
    if len(good_coverage) > 0:
        print(f"[PASS] {len(good_coverage)}/{len(df_results)} d values have >95% coverage")
    else:
        print("[WARN] Low data coverage due to FFD window")

    print()
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
