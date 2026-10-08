"""
Unit tests for quantfund risk controls module.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from quantfund.risk.controls import (
    annualization_factor,
    target_bar_sigma,
    per_trade_weight_cap,
    portfolio_vol_proxy,
    compute_drawdown,
    daily_loss_stopped,
)


class TestAnnualizationFactors:
    """Test suite for annualization factors."""
    
    def test_daily_annualization_factor(self):
        """Test daily interval annualization factor."""
        factor = annualization_factor("1d")
        assert factor == 252.0
    
    def test_hourly_annualization_factor(self):
        """Test 60m interval annualization factor."""
        factor = annualization_factor("60m")
        expected = 252.0 * 6.5  # 6.5 hours trading day
        assert factor == expected
    
    def test_2hour_annualization_factor(self):
        """Test 120m interval annualization factor."""
        factor = annualization_factor("120m")
        expected = 252.0 * 3.25  # 3.25 two-hour periods per day
        assert factor == expected
    
    def test_target_bar_sigma_calculation(self):
        """Test target bar sigma calculation."""
        annual_vol = 0.16  # 16% annual volatility
        
        daily_sigma = target_bar_sigma(annual_vol, "1d")
        expected_daily = annual_vol / np.sqrt(252.0)
        assert abs(daily_sigma - expected_daily) < 1e-10
        
        hourly_sigma = target_bar_sigma(annual_vol, "60m")
        expected_hourly = annual_vol / np.sqrt(252.0 * 6.5)
        assert abs(hourly_sigma - expected_hourly) < 1e-10
    
    @given(annual_vol=st.floats(min_value=0.01, max_value=1.0))
    def test_target_bar_sigma_properties(self, annual_vol):
        """Property-based test for target bar sigma."""
        daily_sigma = target_bar_sigma(annual_vol, "1d")
        hourly_sigma = target_bar_sigma(annual_vol, "60m")
        
        # Hourly sigma should be smaller than daily sigma
        assert hourly_sigma < daily_sigma
        assert daily_sigma > 0
        assert hourly_sigma > 0


class TestPerTradeWeightCap:
    """Test suite for per-trade weight capping."""
    
    def test_basic_weight_cap_calculation(self):
        """Test basic weight cap calculation."""
        asset_vol = 0.02  # 2% volatility
        per_trade_risk = 0.005  # 0.5% risk per trade
        
        cap = per_trade_weight_cap(asset_vol, per_trade_risk)
        expected = per_trade_risk / asset_vol  # 0.25
        
        assert abs(cap - expected) < 1e-10
    
    def test_weight_cap_with_minimum_vol(self):
        """Test weight cap with very low volatility (minimum vol floor)."""
        very_low_vol = 1e-6
        per_trade_risk = 0.01
        min_vol = 1e-4

        cap = per_trade_weight_cap(very_low_vol, per_trade_risk, min_vol)
        # Result should be clipped to 1.0 since per_trade_risk/min_vol = 100 > 1
        assert cap == 1.0
    
    def test_weight_cap_clipping(self):
        """Test that weight cap is clipped to [0, 1]."""
        # Very low volatility should result in cap being clipped to 1.0
        low_vol = 0.001
        high_risk = 0.5
        
        cap = per_trade_weight_cap(low_vol, high_risk)
        assert cap == 1.0
        
        # Negative inputs should result in 0
        cap_negative = per_trade_weight_cap(-0.01, 0.01)
        assert cap_negative == 0.0
    
    @given(
        asset_vol=st.floats(min_value=1e-4, max_value=0.1),
        per_trade_risk=st.floats(min_value=1e-4, max_value=0.1)
    )
    def test_weight_cap_properties(self, asset_vol, per_trade_risk):
        """Property-based test for weight cap."""
        cap = per_trade_weight_cap(asset_vol, per_trade_risk)
        
        assert 0.0 <= cap <= 1.0
        assert isinstance(cap, float)


class TestPortfolioVolProxy:
    """Test suite for portfolio volatility proxy calculation."""
    
    def test_single_asset_portfolio(self):
        """Test portfolio vol with single asset."""
        weights = np.array([1.0])
        vols = np.array([0.02])
        
        port_vol = portfolio_vol_proxy(weights, vols)
        expected = np.sqrt((1.0 * 0.02) ** 2)
        
        assert abs(port_vol - expected) < 1e-10
    
    def test_equal_weight_portfolio(self):
        """Test portfolio vol with equal weights."""
        weights = np.array([0.5, 0.5])
        vols = np.array([0.02, 0.03])
        
        port_vol = portfolio_vol_proxy(weights, vols)
        expected = np.sqrt((0.5 * 0.02) ** 2 + (0.5 * 0.03) ** 2)
        
        assert abs(port_vol - expected) < 1e-10
    
    def test_zero_weights_portfolio(self):
        """Test portfolio vol with zero weights."""
        weights = np.array([0.0, 0.0, 0.0])
        vols = np.array([0.02, 0.03, 0.01])
        
        port_vol = portfolio_vol_proxy(weights, vols)
        assert port_vol == 0.0
    
    def test_portfolio_vol_scaling(self):
        """Test that doubling weights doubles portfolio vol."""
        weights = np.array([0.3, 0.4, 0.3])
        vols = np.array([0.02, 0.03, 0.015])
        
        vol1 = portfolio_vol_proxy(weights, vols)
        vol2 = portfolio_vol_proxy(2 * weights, vols)
        
        assert abs(vol2 - 2 * vol1) < 1e-10
    
    @given(
        weights=st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=10),
        vols=st.lists(st.floats(min_value=1e-4, max_value=0.1), min_size=1, max_size=10)
    )
    def test_portfolio_vol_properties(self, weights, vols):
        """Property-based test for portfolio vol proxy."""
        if len(weights) != len(vols):
            weights = weights[:min(len(weights), len(vols))]
            vols = vols[:min(len(weights), len(vols))]
        
        if not weights:  # Skip empty lists
            return
            
        w_arr = np.array(weights)
        v_arr = np.array(vols)
        
        port_vol = portfolio_vol_proxy(w_arr, v_arr)
        
        assert port_vol >= 0.0
        assert isinstance(port_vol, float)


class TestDrawdownComputation:
    """Test suite for drawdown computation."""
    
    def test_no_drawdown_increasing_equity(self):
        """Test drawdown with always increasing equity."""
        equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        dd = compute_drawdown(equity)
        
        # No drawdown for monotonically increasing equity
        assert (dd <= 0.0).all()
        assert (dd >= -1e-10).all()  # Should be essentially zero
    
    def test_simple_drawdown_calculation(self):
        """Test drawdown calculation with simple pattern."""
        equity = pd.Series([1.0, 1.2, 1.0, 0.8, 1.1])
        dd = compute_drawdown(equity)
        
        expected = pd.Series([0.0, 0.0, -1/6, -1/3, -1/12])  # Relative to peaks
        
        for i, (actual, exp) in enumerate(zip(dd, expected)):
            assert abs(actual - exp) < 1e-10, f"Mismatch at index {i}"
    
    def test_maximum_drawdown_identification(self):
        """Test identification of maximum drawdown."""
        equity = pd.Series([1.0, 1.5, 1.2, 0.6, 0.9, 1.4])
        dd = compute_drawdown(equity)
        
        max_dd = dd.min()
        # From peak of 1.5 to trough of 0.6 = -60%
        expected_max_dd = (0.6 / 1.5) - 1.0
        
        assert abs(max_dd - expected_max_dd) < 1e-10
    
    def test_empty_equity_series(self):
        """Test drawdown with empty equity series."""
        empty_equity = pd.Series(dtype=float)
        dd = compute_drawdown(empty_equity)
        
        assert len(dd) == 0
        assert dd.dtype == float
    
    def test_single_value_equity(self):
        """Test drawdown with single equity value."""
        single_equity = pd.Series([1.0])
        dd = compute_drawdown(single_equity)
        
        assert len(dd) == 1
        assert dd.iloc[0] == 0.0


class TestDailyLossStop:
    """Test suite for daily loss stop mechanism."""
    
    def test_no_loss_stop_triggered(self):
        """Test when no loss stop should be triggered."""
        dates = pd.date_range('2023-01-01 09:30', periods=10, freq='1H')
        returns = pd.Series([0.01, 0.005, -0.002, 0.003, -0.001] * 2, index=dates)
        
        stop_signal = daily_loss_stopped(returns, threshold=0.05)
        
        assert not stop_signal.any(), "No stop should be triggered with small losses"
    
    def test_loss_stop_triggered_same_day(self):
        """Test loss stop triggered within same trading day."""
        dates = pd.date_range('2023-01-01 09:30', periods=6, freq='1H')
        # Cumulative: 0.01, 0.01, -0.02, -0.04, -0.06 (triggers at index 4)
        returns = pd.Series([0.01, 0.00, -0.03, -0.02, -0.02, 0.01], index=dates)
        
        stop_signal = daily_loss_stopped(returns, threshold=0.05)
        
        # Should trigger at index 4 (-6% cumulative) and remain True for rest of day
        expected = [False, False, False, False, True, True]
        
        for i, (actual, exp) in enumerate(zip(stop_signal, expected)):
            assert actual == exp, f"Mismatch at index {i}"
    
    def test_loss_stop_multiple_days(self):
        """Test loss stop across multiple days."""
        # Day 1: 9:30-14:30, Day 2: 9:30-14:30
        dates1 = pd.date_range('2023-01-01 09:30', periods=5, freq='1H')
        dates2 = pd.date_range('2023-01-02 09:30', periods=5, freq='1H')
        all_dates = dates1.append(dates2)
        
        # Day 1: trigger stop, Day 2: normal trading
        returns = pd.Series([0.01, -0.02, -0.03, -0.01, 0.01,  # Day 1
                            0.02, 0.01, -0.01, 0.005, 0.01], index=all_dates)  # Day 2
        
        stop_signal = daily_loss_stopped(returns, threshold=0.04)
        
        # Day 1: stop triggered at index 2 (-4% cumulative)
        # Day 2: fresh start, no stop
        expected = [False, False, True, True, True,  # Day 1
                   False, False, False, False, False]  # Day 2
        
        for i, (actual, exp) in enumerate(zip(stop_signal, expected)):
            assert actual == exp, f"Mismatch at index {i}"
    
    def test_empty_returns_series(self):
        """Test daily loss stop with empty returns."""
        empty_returns = pd.Series(dtype=float)
        stop_signal = daily_loss_stopped(empty_returns, threshold=0.05)
        
        assert len(stop_signal) == 0
        assert stop_signal.dtype == bool
    
    def test_loss_stop_edge_cases(self):
        """Test edge cases for loss stop."""
        dates = pd.date_range('2023-01-01 09:30', periods=3, freq='1H')
        
        # Exactly at threshold
        returns_exact = pd.Series([0.0, -0.025, -0.025], index=dates)
        stop_exact = daily_loss_stopped(returns_exact, threshold=0.05)
        assert stop_exact.iloc[2], "Should trigger when exactly at threshold"
        
        # Just below threshold
        returns_below = pd.Series([0.0, -0.024, -0.025], index=dates)
        stop_below = daily_loss_stopped(returns_below, threshold=0.05)
        assert not stop_below.any(), "Should not trigger when just below threshold"
    
    @given(
        threshold=st.floats(min_value=0.01, max_value=0.2),
        loss_magnitude=st.floats(min_value=0.01, max_value=0.3)
    )
    def test_loss_stop_properties(self, threshold, loss_magnitude):
        """Property-based test for loss stop mechanism."""
        dates = pd.date_range('2023-01-01 09:30', periods=5, freq='1H')
        
        if loss_magnitude >= threshold:
            # Create a return series that will trigger the stop
            returns = pd.Series([0.0, -loss_magnitude/2, -loss_magnitude/2, 0.01, 0.01], index=dates)
            stop_signal = daily_loss_stopped(returns, threshold)

            # Once triggered, should remain True for rest of day
            if stop_signal.any():
                first_stop = stop_signal.idxmax()
                after_stop = stop_signal.loc[first_stop:]
                assert after_stop.all(), "Stop should remain active after trigger"
        else:
            # Loss not large enough to trigger
            returns = pd.Series([0.0, -loss_magnitude, 0.01, 0.01, 0.01], index=dates)
            stop_signal = daily_loss_stopped(returns, threshold)

            assert not stop_signal.any(), "Stop should not trigger for small losses"


class TestRiskControlsIntegration:
    """Integration tests for risk control components."""
    
    def test_risk_controls_workflow(self):
        """Test typical risk controls workflow."""
        # Setup portfolio parameters
        annual_vol_target = 0.12
        per_trade_risk = 0.005
        
        # Asset parameters
        weights = np.array([0.4, 0.3, 0.3])
        asset_vols = np.array([0.02, 0.025, 0.018])
        
        # Calculate target volatility for daily interval
        target_vol = target_bar_sigma(annual_vol_target, "1d")
        
        # Calculate individual asset caps
        caps = [per_trade_weight_cap(vol, per_trade_risk) for vol in asset_vols]
        
        # Calculate portfolio vol
        port_vol = portfolio_vol_proxy(weights, asset_vols)
        
        # All components should work together
        assert target_vol > 0
        assert all(cap > 0 for cap in caps)
        assert port_vol > 0
        
        # Portfolio vol should be less than sum of individual contributions
        individual_sum = sum(w * v for w, v in zip(weights, asset_vols))
        assert port_vol <= individual_sum
    
    def test_risk_controls_scaling_consistency(self):
        """Test that risk controls scale consistently."""
        weights = np.array([0.5, 0.5])
        vols = np.array([0.02, 0.03])
        
        # Original portfolio vol
        vol1 = portfolio_vol_proxy(weights, vols)
        
        # Scaled weights
        vol2 = portfolio_vol_proxy(2 * weights, vols)
        
        # Should scale linearly
        assert abs(vol2 - 2 * vol1) < 1e-10