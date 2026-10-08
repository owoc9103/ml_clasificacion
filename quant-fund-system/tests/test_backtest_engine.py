"""
Unit tests for quantfund backtest engine.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from quantfund.backtest.engine import (
    CostModel,
    RiskLimits,
    BacktestConfig,
    backtest_signals,
    _safe_atr,
)


class TestCostModel:
    """Test suite for cost model."""
    
    def test_default_cost_model(self):
        """Test default cost model parameters."""
        cost = CostModel()
        
        assert cost.commission_bps == 2.0
        assert cost.slippage_bps == 1.0
        assert cost.roundtrip_cost_rate == 3.0 * 1e-4  # 3 bps total
    
    def test_custom_cost_model(self):
        """Test custom cost model parameters."""
        cost = CostModel(commission_bps=1.5, slippage_bps=0.5)
        
        assert cost.commission_bps == 1.5
        assert cost.slippage_bps == 0.5
        assert cost.roundtrip_cost_rate == 2.0 * 1e-4  # 2 bps total
    
    def test_cost_model_immutability(self):
        """Test that cost model is immutable (frozen dataclass)."""
        cost = CostModel()
        
        with pytest.raises(AttributeError):
            cost.commission_bps = 5.0


class TestRiskLimits:
    """Test suite for risk limits."""
    
    def test_default_risk_limits(self):
        """Test default risk limit parameters."""
        risk = RiskLimits()
        
        assert risk.annual_vol_target == 0.10
        assert risk.max_drawdown == 0.12
        assert risk.per_trade_risk == 0.005
        assert risk.daily_loss_stop == 0.01
        assert risk.atr_stop_mult == 3.0
        assert risk.kelly_cap == 0.15
    
    def test_custom_risk_limits(self):
        """Test custom risk limit parameters."""
        risk = RiskLimits(
            annual_vol_target=0.15,
            max_drawdown=0.20,
            per_trade_risk=0.01,
            kelly_cap=0.25
        )
        
        assert risk.annual_vol_target == 0.15
        assert risk.max_drawdown == 0.20
        assert risk.per_trade_risk == 0.01
        assert risk.kelly_cap == 0.25
    
    def test_risk_limits_immutability(self):
        """Test that risk limits are immutable."""
        risk = RiskLimits()
        
        with pytest.raises(AttributeError):
            risk.annual_vol_target = 0.20


class TestBacktestConfig:
    """Test suite for backtest configuration."""
    
    def test_default_backtest_config(self):
        """Test default backtest configuration."""
        config = BacktestConfig(
            interval="1d",
            cost=CostModel(),
            risk=RiskLimits()
        )
        
        assert config.interval == "1d"
        assert isinstance(config.cost, CostModel)
        assert isinstance(config.risk, RiskLimits)
    
    def test_backtest_config_different_intervals(self):
        """Test backtest config with different intervals."""
        intervals = ["1d", "60m", "120m"]
        
        for interval in intervals:
            config = BacktestConfig(
                interval=interval,
                cost=CostModel(),
                risk=RiskLimits()
            )
            assert config.interval == interval


class TestSafeATR:
    """Test suite for safe ATR calculation."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample OHLC price data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='1D')
        
        # Generate realistic price series
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50)))
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, 50)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, 50)))
        
        return {
            'high': pd.Series(high_prices, index=dates),
            'low': pd.Series(low_prices, index=dates),
            'close': pd.Series(close_prices, index=dates)
        }
    
    def test_atr_basic_calculation(self, sample_price_data):
        """Test basic ATR calculation."""
        atr = _safe_atr(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            window=14
        )
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_price_data['close'])
        assert (atr >= 0).all(), "ATR should always be non-negative"
        assert atr.notna().all(), "ATR should not have NaN values"
    
    def test_atr_empty_series(self):
        """Test ATR with empty price series."""
        empty_series = pd.Series(dtype=float)
        
        atr = _safe_atr(empty_series, empty_series, empty_series)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == 0
        assert atr.dtype == float
    
    def test_atr_single_value(self):
        """Test ATR with single price value."""
        single_idx = pd.date_range('2023-01-01', periods=1)
        single_price = pd.Series([100.0], index=single_idx)
        
        atr = _safe_atr(single_price, single_price, single_price)
        
        assert len(atr) == 1
        assert atr.iloc[0] == 0.0  # No true range for single value
    
    def test_atr_constant_prices(self):
        """Test ATR with constant prices (no volatility)."""
        dates = pd.date_range('2023-01-01', periods=20, freq='1D')
        constant_price = pd.Series([100.0] * 20, index=dates)
        
        atr = _safe_atr(constant_price, constant_price, constant_price)
        
        assert (atr == 0.0).all(), "ATR should be zero for constant prices"
    
    def test_atr_different_windows(self, sample_price_data):
        """Test ATR with different window sizes."""
        windows = [5, 10, 14, 20]
        
        for window in windows:
            atr = _safe_atr(
                sample_price_data['high'],
                sample_price_data['low'],
                sample_price_data['close'],
                window=window
            )
            
            assert len(atr) == len(sample_price_data['close'])
            assert (atr >= 0).all()


class TestBacktestSignals:
    """Test suite for main backtest function."""
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample data for backtesting."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        np.random.seed(42)
        
        symbols = ['SPY', 'QQQ']
        prices_by_symbol = {}
        signals_by_symbol = {}
        
        for symbol in symbols:
            # Generate realistic price data
            returns = np.random.normal(0.0005, 0.01, 100)  # Slight positive drift
            close_prices = 100 * np.exp(np.cumsum(returns))
            
            prices_by_symbol[symbol] = pd.DataFrame({
                'open': close_prices * (1 + np.random.normal(0, 0.001, 100)),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
                'close': close_prices
            }, index=dates)
            
            # Generate random signals
            signals_by_symbol[symbol] = pd.Series(
                np.random.uniform(0, 0.5, 100), 
                index=dates,
                name=f'{symbol}_signal'
            )
        
        return prices_by_symbol, signals_by_symbol
    
    @pytest.fixture
    def sample_config(self):
        """Create sample backtest configuration."""
        return BacktestConfig(
            interval="1d",
            cost=CostModel(commission_bps=2.0, slippage_bps=1.0),
            risk=RiskLimits(
                annual_vol_target=0.10,
                max_drawdown=0.15,
                per_trade_risk=0.005,
                daily_loss_stop=0.02
            )
        )
    
    def test_backtest_basic_functionality(self, sample_backtest_data, sample_config):
        """Test basic backtest functionality."""
        prices_by_symbol, signals_by_symbol = sample_backtest_data
        
        results = backtest_signals(prices_by_symbol, signals_by_symbol, sample_config)
        
        # Check return structure
        assert isinstance(results, dict)
        assert 'timeseries' in results
        assert 'trades' in results
        assert 'perf' in results
        
        # Check timeseries data
        ts = results['timeseries']
        assert isinstance(ts, pd.DataFrame)
        assert 'portfolio_ret' in ts.columns
        assert 'equity' in ts.columns
        assert 'drawdown' in ts.columns
        
        # Check performance data
        perf = results['perf']
        assert isinstance(perf, pd.DataFrame)
        expected_cols = ['ann_return', 'ann_vol', 'sharpe', 'max_drawdown', 'num_trades']
        for col in expected_cols:
            assert col in perf.columns
    
    def test_backtest_empty_input(self, sample_config):
        """Test backtest with empty input data."""
        empty_prices = {}
        empty_signals = {}
        
        results = backtest_signals(empty_prices, empty_signals, sample_config)
        
        assert results['timeseries'].empty
        assert results['trades'].empty
        assert results['perf'].empty
    
    def test_backtest_single_symbol(self, sample_config):
        """Test backtest with single symbol."""
        dates = pd.date_range('2023-01-01', periods=50, freq='1D')
        np.random.seed(42)
        
        symbol = 'SPY'
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50)))
        
        prices = {
            symbol: pd.DataFrame({
                'open': close_prices,
                'high': close_prices * 1.01,
                'low': close_prices * 0.99,
                'close': close_prices
            }, index=dates)
        }
        
        signals = {
            symbol: pd.Series(0.2, index=dates)  # Constant signal
        }
        
        results = backtest_signals(prices, signals, sample_config)
        
        # Should produce valid results
        assert not results['timeseries'].empty
        assert len(results['perf']) == 1
    
    def test_backtest_zero_signals(self, sample_backtest_data, sample_config):
        """Test backtest with zero signals (no trading)."""
        prices_by_symbol, _ = sample_backtest_data
        
        # Zero signals for all symbols
        zero_signals = {
            symbol: pd.Series(0.0, index=prices.index)
            for symbol, prices in prices_by_symbol.items()
        }
        
        results = backtest_signals(prices_by_symbol, zero_signals, sample_config)
        
        # Should have zero returns and no trades
        ts = results['timeseries']
        assert (ts['portfolio_ret'].fillna(0) == 0).all()
        assert results['trades'].empty or len(results['trades']) == 0
    
    def test_backtest_high_signals(self, sample_backtest_data, sample_config):
        """Test backtest with high signals (maximum allocation)."""
        prices_by_symbol, _ = sample_backtest_data
        
        # Maximum signals for all symbols
        high_signals = {
            symbol: pd.Series(1.0, index=prices.index)
            for symbol, prices in prices_by_symbol.items()
        }
        
        results = backtest_signals(prices_by_symbol, high_signals, sample_config)
        
        # Should have active trading and non-zero returns
        ts = results['timeseries']
        assert not (ts['portfolio_ret'].fillna(0) == 0).all()
        assert not results['trades'].empty
    
    def test_backtest_cost_impact(self, sample_backtest_data):
        """Test that transaction costs reduce returns."""
        prices_by_symbol, signals_by_symbol = sample_backtest_data
        
        # High cost configuration
        high_cost_config = BacktestConfig(
            interval="1d",
            cost=CostModel(commission_bps=10.0, slippage_bps=5.0),  # 15 bps total
            risk=RiskLimits()
        )
        
        # Low cost configuration
        low_cost_config = BacktestConfig(
            interval="1d",
            cost=CostModel(commission_bps=1.0, slippage_bps=0.5),  # 1.5 bps total
            risk=RiskLimits()
        )
        
        high_cost_results = backtest_signals(prices_by_symbol, signals_by_symbol, high_cost_config)
        low_cost_results = backtest_signals(prices_by_symbol, signals_by_symbol, low_cost_config)
        
        # High cost should result in lower returns (assuming same gross performance)
        high_cost_ret = high_cost_results['perf']['ann_return'].iloc[0]
        low_cost_ret = low_cost_results['perf']['ann_return'].iloc[0]
        
        assert high_cost_ret <= low_cost_ret, "Higher costs should reduce returns"
    
    def test_backtest_risk_controls(self, sample_backtest_data):
        """Test that risk controls are enforced."""
        prices_by_symbol, signals_by_symbol = sample_backtest_data
        
        # Strict risk limits
        strict_config = BacktestConfig(
            interval="1d",
            cost=CostModel(),
            risk=RiskLimits(
                annual_vol_target=0.05,  # Very low vol target
                max_drawdown=0.05,       # Very low drawdown limit
                per_trade_risk=0.001,    # Very low per-trade risk
                daily_loss_stop=0.005    # Very low daily loss stop
            )
        )
        
        results = backtest_signals(prices_by_symbol, signals_by_symbol, strict_config)
        
        # Check that risk limits are respected
        ts = results['timeseries']
        perf = results['perf']
        
        # Maximum drawdown should be within limits (may be slightly higher due to discrete trading)
        max_dd = abs(perf['max_drawdown'].iloc[0])
        assert max_dd <= 0.10, "Drawdown should be controlled by risk limits"  # Allow some tolerance
        
        # Volatility should be controlled
        ann_vol = perf['ann_vol'].iloc[0]
        assert ann_vol <= 0.15, "Volatility should be controlled by risk limits"  # Allow some tolerance
    
    def test_backtest_equity_curve_properties(self, sample_backtest_data, sample_config):
        """Test properties of the equity curve."""
        prices_by_symbol, signals_by_symbol = sample_backtest_data
        
        results = backtest_signals(prices_by_symbol, signals_by_symbol, sample_config)
        
        ts = results['timeseries']
        equity = ts['equity']
        
        # Equity should start at 1.0
        assert abs(equity.iloc[0] - 1.0) < 1e-10
        
        # Equity should always be positive
        assert (equity > 0).all()
        
        # Drawdown should always be <= 0
        drawdown = ts['drawdown']
        assert (drawdown <= 0).all()
        
        # Drawdown should start at 0
        assert abs(drawdown.iloc[0]) < 1e-10
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        vol_target=st.floats(min_value=0.05, max_value=0.25),
        max_dd=st.floats(min_value=0.05, max_value=0.30)
    )
    def test_backtest_different_risk_parameters(self, sample_backtest_data, vol_target, max_dd):
        """Property-based test with different risk parameters."""
        prices_by_symbol, signals_by_symbol = sample_backtest_data
        
        config = BacktestConfig(
            interval="1d",
            cost=CostModel(),
            risk=RiskLimits(
                annual_vol_target=vol_target,
                max_drawdown=max_dd,
                per_trade_risk=0.005,
                daily_loss_stop=0.02
            )
        )
        
        results = backtest_signals(prices_by_symbol, signals_by_symbol, config)
        
        # Basic invariants should hold
        assert isinstance(results, dict)
        assert 'timeseries' in results
        assert 'trades' in results
        assert 'perf' in results
        
        if not results['timeseries'].empty:
            equity = results['timeseries']['equity']
            assert (equity > 0).all()
            assert abs(equity.iloc[0] - 1.0) < 1e-10


class TestBacktestIntegration:
    """Integration tests for backtest components."""
    
    def test_backtest_with_realistic_strategy(self):
        """Test backtest with realistic strategy implementation."""
        # Create trending market data
        dates = pd.date_range('2023-01-01', periods=252, freq='1D')  # One year
        np.random.seed(42)
        
        symbols = ['SPY', 'QQQ', 'IWM']
        prices_by_symbol = {}
        signals_by_symbol = {}
        
        for i, symbol in enumerate(symbols):
            # Different trend strengths for each symbol
            trend = 0.0003 * (i + 1)  # SPY: 0.03%, QQQ: 0.06%, IWM: 0.09% daily
            volatility = 0.01 + 0.002 * i  # Increasing volatility
            
            returns = np.random.normal(trend, volatility, 252)
            close_prices = 100 * np.exp(np.cumsum(returns))
            
            prices_by_symbol[symbol] = pd.DataFrame({
                'open': close_prices * (1 + np.random.normal(0, 0.0005, 252)),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.001, 252))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.001, 252))),
                'close': close_prices
            }, index=dates)
            
            # Momentum-based signals (higher for trending assets)
            momentum_score = np.minimum(0.8, np.maximum(0.0, 
                0.2 + 0.3 * (trend / 0.0003)))  # Scale with trend strength
            
            signals_by_symbol[symbol] = pd.Series(
                momentum_score + np.random.normal(0, 0.1, 252),
                index=dates
            ).clip(0, 1)
        
        config = BacktestConfig(
            interval="1d",
            cost=CostModel(commission_bps=2.0, slippage_bps=1.0),
            risk=RiskLimits(
                annual_vol_target=0.12,
                max_drawdown=0.15,
                per_trade_risk=0.005,
                daily_loss_stop=0.02
            )
        )
        
        results = backtest_signals(prices_by_symbol, signals_by_symbol, config)
        
        # Realistic expectations for trending market with momentum strategy
        perf = results['perf']
        ts = results['timeseries']
        
        # Should have positive returns in trending market
        ann_return = perf['ann_return'].iloc[0]
        assert ann_return > -0.10, "Should not have large negative returns in trending market"
        
        # Should have reasonable Sharpe ratio
        sharpe = perf['sharpe'].iloc[0]
        assert sharpe > -2.0, "Sharpe ratio should be reasonable"
        
        # Should have controlled volatility
        ann_vol = perf['ann_vol'].iloc[0]
        assert ann_vol < 0.25, "Volatility should be controlled"
        
        # Should have some trading activity
        num_trades = perf['num_trades'].iloc[0]
        assert num_trades > 0, "Should have some trading activity"
        
        # Equity curve should be reasonable
        final_equity = ts['equity'].iloc[-1]
        assert final_equity > 0.5, "Final equity should be reasonable"
        assert final_equity < 3.0, "Final equity should not be unrealistically high"