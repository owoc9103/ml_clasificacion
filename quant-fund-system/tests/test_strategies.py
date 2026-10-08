"""
Unit tests for quantfund strategies module.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from quantfund.strategies.momo import momo_signals
from quantfund.strategies.meanrev import meanrev_signals


class TestMomentumStrategy:
    """Test suite for momentum strategy signals."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_proba(self, sample_data):
        """Create sample probability predictions."""
        np.random.seed(42)
        return pd.Series(
            np.random.beta(2, 2, len(sample_data)), 
            index=sample_data.index,
            name='p_up'
        )
    
    @pytest.fixture
    def sample_regime(self, sample_data):
        """Create sample regime filter."""
        np.random.seed(42)
        return pd.Series(
            np.random.choice([True, False], len(sample_data), p=[0.7, 0.3]),
            index=sample_data.index,
            name='regime_ok'
        )
    
    def test_momo_signals_basic_functionality(self, sample_data, sample_proba, sample_regime):
        """Test basic momentum signal generation."""
        signals = momo_signals(sample_data, sample_proba, sample_regime)
        
        assert isinstance(signals, pd.Series)
        assert signals.name == "momo_signal"
        assert len(signals) == len(sample_data)
        assert signals.index.equals(sample_data.index)
    
    def test_momo_signals_range_constraints(self, sample_data, sample_proba, sample_regime):
        """Test that momentum signals are in valid range [0, 1]."""
        signals = momo_signals(sample_data, sample_proba, sample_regime)
        
        assert (signals >= 0.0).all(), "All signals should be non-negative"
        assert (signals <= 1.0).all(), "All signals should be <= 1.0"
    
    def test_momo_signals_empty_dataframe(self):
        """Test momentum signals with empty input."""
        empty_df = pd.DataFrame()
        empty_proba = pd.Series(dtype=float)
        empty_regime = pd.Series(dtype=bool)
        
        signals = momo_signals(empty_df, empty_proba, empty_regime)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == 0
        assert signals.dtype == float
    
    def test_momo_signals_regime_filter(self, sample_data):
        """Test that regime filter properly zeros out signals."""
        # High probability signal
        high_proba = pd.Series(0.8, index=sample_data.index)
        # All regime bad
        bad_regime = pd.Series(False, index=sample_data.index)
        
        signals = momo_signals(sample_data, high_proba, bad_regime)
        
        assert (signals == 0.0).all(), "All signals should be zero when regime is bad"
    
    def test_momo_signals_probability_threshold(self, sample_data, sample_regime):
        """Test that probabilities below 0.5 produce zero signals."""
        low_proba = pd.Series(0.3, index=sample_data.index)
        
        signals = momo_signals(sample_data, low_proba, sample_regime)
        
        # Where regime is ok, signals should still be zero due to low probability
        regime_ok_mask = sample_regime.reindex(sample_data.index).fillna(False)
        zero_signals = signals[regime_ok_mask]
        assert (zero_signals == 0.0).all(), "Low probability should produce zero signals"
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        proba_val=st.floats(min_value=0.5, max_value=1.0),
        regime_val=st.booleans()
    )
    def test_momo_signals_property_based(self, sample_data, proba_val, regime_val):
        """Property-based test for momentum signals."""
        proba = pd.Series(proba_val, index=sample_data.index)
        regime = pd.Series(regime_val, index=sample_data.index)
        
        signals = momo_signals(sample_data, proba, regime)
        
        # Properties that should always hold
        assert (signals >= 0.0).all()
        assert (signals <= 1.0).all()

        if regime_val and proba_val > 0.5:
            # When regime is good and prob > 0.5, signal should be positive
            assert (signals > 0.0).all()
        elif not regime_val or proba_val == 0.5:
            # When regime is bad or prob exactly 0.5, signal should be zero
            assert (signals == 0.0).all()


class TestMeanReversionStrategy:
    """Test suite for mean reversion strategy signals."""
    
    @pytest.fixture
    def sample_data_with_indicators(self):
        """Create sample data with technical indicators."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # Create realistic z-score and RSI data
        zscore = np.random.normal(0, 1, 100)
        rsi = np.random.uniform(20, 80, 100)
        
        return pd.DataFrame({
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'zscore_20': zscore,
            'rsi_14': rsi,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_regime(self, sample_data_with_indicators):
        """Create sample regime filter."""
        np.random.seed(42)
        return pd.Series(
            np.random.choice([True, False], len(sample_data_with_indicators), p=[0.8, 0.2]),
            index=sample_data_with_indicators.index,
            name='regime_ok'
        )
    
    def test_meanrev_signals_basic_functionality(self, sample_data_with_indicators, sample_regime):
        """Test basic mean reversion signal generation."""
        signals = meanrev_signals(sample_data_with_indicators, sample_regime)
        
        assert isinstance(signals, pd.Series)
        assert signals.name == "meanrev_signal"
        assert len(signals) == len(sample_data_with_indicators)
        assert signals.index.equals(sample_data_with_indicators.index)
    
    def test_meanrev_signals_range_constraints(self, sample_data_with_indicators, sample_regime):
        """Test that mean reversion signals are in valid range [0, 1]."""
        signals = meanrev_signals(sample_data_with_indicators, sample_regime)
        
        assert (signals >= 0.0).all(), "All signals should be non-negative"
        assert (signals <= 1.0).all(), "All signals should be <= 1.0"
    
    def test_meanrev_signals_empty_dataframe(self):
        """Test mean reversion signals with empty input."""
        empty_df = pd.DataFrame()
        empty_regime = pd.Series(dtype=bool)
        
        signals = meanrev_signals(empty_df, empty_regime)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == 0
        assert signals.dtype == float
    
    def test_meanrev_signals_missing_indicators(self, sample_regime):
        """Test behavior when required indicators are missing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        df_no_indicators = pd.DataFrame({
            'close': np.random.uniform(95, 105, 10)
        }, index=dates)
        
        signals = meanrev_signals(df_no_indicators, sample_regime[:10])
        
        # Should return zeros when indicators are missing
        assert (signals == 0.0).all()
        assert signals.name == "meanrev_signal"
    
    def test_meanrev_signals_regime_filter(self, sample_data_with_indicators):
        """Test that regime filter properly zeros out signals."""
        # Create conditions for strong mean reversion signal
        strong_meanrev_data = sample_data_with_indicators.copy()
        strong_meanrev_data['zscore_20'] = -2.5  # Strong negative z-score
        strong_meanrev_data['rsi_14'] = 25       # Oversold RSI
        
        # All regime bad
        bad_regime = pd.Series(False, index=strong_meanrev_data.index)
        
        signals = meanrev_signals(strong_meanrev_data, bad_regime)
        
        assert (signals == 0.0).all(), "All signals should be zero when regime is bad"
    
    def test_meanrev_signals_strong_contrarian_conditions(self, sample_regime):
        """Test signal strength under strong contrarian conditions."""
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        strong_data = pd.DataFrame({
            'close': np.random.uniform(95, 105, 10),
            'zscore_20': np.full(10, -3.0),  # Maximum negative z-score (clipped)
            'rsi_14': np.full(10, 20),       # Very oversold
        }, index=dates)
        
        # All regime good
        good_regime = pd.Series(True, index=strong_data.index)
        
        signals = meanrev_signals(strong_data, good_regime)
        
        # Should produce strong signals
        assert (signals > 0.5).all(), "Strong contrarian conditions should produce strong signals"
    
    def test_meanrev_signals_alternative_zscore_column(self, sample_regime):
        """Test that function handles alternative z-score column name."""
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        alt_data = pd.DataFrame({
            'close': np.random.uniform(95, 105, 10),
            'ema20_z': np.random.normal(0, 1, 10),  # Alternative column name
            'rsi_14': np.random.uniform(30, 70, 10),
        }, index=dates)
        
        signals = meanrev_signals(alt_data, sample_regime[:10])
        
        # Should work with alternative column name
        assert isinstance(signals, pd.Series)
        assert len(signals) == 10
        assert not (signals == 0.0).all()  # Should have some non-zero signals
    
    @given(
        zscore_val=st.floats(min_value=-3.0, max_value=3.0),
        rsi_val=st.floats(min_value=0.0, max_value=100.0),
        regime_val=st.booleans()
    )
    def test_meanrev_signals_property_based(self, zscore_val, rsi_val, regime_val):
        """Property-based test for mean reversion signals."""
        dates = pd.date_range('2023-01-01', periods=5, freq='1H')
        data = pd.DataFrame({
            'close': [100] * 5,
            'zscore_20': [zscore_val] * 5,
            'rsi_14': [rsi_val] * 5,
        }, index=dates)
        regime = pd.Series([regime_val] * 5, index=dates)
        
        signals = meanrev_signals(data, regime)
        
        # Properties that should always hold
        assert (signals >= 0.0).all()
        assert (signals <= 1.0).all()
        
        if not regime_val:
            # When regime is bad, signal should be zero
            assert (signals == 0.0).all()


class TestStrategiesIntegration:
    """Integration tests for strategy components."""
    
    def test_strategies_complementary_signals(self):
        """Test that momentum and mean reversion can produce complementary signals."""
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')
        
        # Create trending market (good for momentum)
        trending_data = pd.DataFrame({
            'close': np.linspace(100, 120, 50),
            'zscore_20': np.random.normal(0, 0.5, 50),
            'rsi_14': np.random.uniform(40, 80, 50),
        }, index=dates)
        
        # High momentum probability
        high_proba = pd.Series(0.8, index=dates)
        good_regime = pd.Series(True, index=dates)
        
        momo_sigs = momo_signals(trending_data, high_proba, good_regime)
        meanrev_sigs = meanrev_signals(trending_data, good_regime)
        
        # In trending markets, momentum should generally be stronger
        assert momo_sigs.mean() > 0.2, "Momentum should be active in trending market"
        
        # Both strategies should respect the same regime filter
        assert not (momo_sigs < 0).any()
        assert not (meanrev_sigs < 0).any()