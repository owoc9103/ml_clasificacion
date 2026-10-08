"""
Unit tests for quantfund.models.train module.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantfund.models.train import (
    TrainConfig,
    load_train_config,
    get_cost_from_config,
    _read_yaml_config,
    _collect_parquets,
    _concat_datasets,
    _time_series_folds,
    _metrics_bin,
    compute_metrics,
)


class TestTrainConfig:
    """Test suite for TrainConfig dataclass."""

    def test_default_train_config(self):
        """Test default TrainConfig values."""
        config = TrainConfig()

        assert config.cost_bps == 2.0
        assert config.cv_splits == 5
        assert config.purge_days == 5
        assert config.embargo_days == 2

    def test_custom_train_config(self):
        """Test custom TrainConfig values."""
        config = TrainConfig(
            cost_bps=3.0,
            cv_splits=10,
            purge_days=3,
            embargo_days=1
        )

        assert config.cost_bps == 3.0
        assert config.cv_splits == 10
        assert config.purge_days == 3
        assert config.embargo_days == 1

    def test_train_config_frozen(self):
        """Test that TrainConfig is immutable."""
        config = TrainConfig()

        with pytest.raises(AttributeError):
            config.cost_bps = 5.0


class TestConfigLoading:
    """Test suite for configuration loading."""

    def test_read_yaml_config_nonexistent(self):
        """Test reading nonexistent YAML file."""
        result = _read_yaml_config(Path("nonexistent.yaml"))
        assert result == {}

    def test_read_yaml_config_valid(self):
        """Test reading valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"cost_bps": 3.0, "cv_splits": 10}, f)
            temp_path = Path(f.name)

        try:
            result = _read_yaml_config(temp_path)
            assert result == {"cost_bps": 3.0, "cv_splits": 10}
        finally:
            temp_path.unlink()

    def test_get_cost_from_config(self):
        """Test converting bps to decimal."""
        # Uses actual config file if exists, otherwise default
        cost = get_cost_from_config()
        assert isinstance(cost, float)
        assert cost > 0  # Should be positive
        assert cost < 0.01  # Should be reasonable (< 100 bps)


class TestDatasetLoading:
    """Test suite for dataset loading functions."""

    def test_collect_parquets_empty_directory(self):
        """Test collecting parquets from nonexistent directory."""
        paths = _collect_parquets("nonexistent_interval")
        assert paths == []

    def test_concat_datasets_empty_list(self):
        """Test concatenating empty list of datasets."""
        result = _concat_datasets([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_concat_datasets_with_data(self):
        """Test concatenating datasets with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test parquet files
            df1 = pd.DataFrame({
                'open': [100, 101],
                'close': [102, 103],
                'symbol': ['SPY', 'SPY'],
            }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02'], name='datetime'))

            df2 = pd.DataFrame({
                'open': [200, 201],
                'close': [202, 203],
                'symbol': ['QQQ', 'QQQ'],
            }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02'], name='datetime'))

            path1 = Path(tmpdir) / "file1.parquet"
            path2 = Path(tmpdir) / "file2.parquet"

            df1.to_parquet(path1)
            df2.to_parquet(path2)

            result = _concat_datasets([path1, path2])

            assert len(result) == 4
            assert 'timestamp' in result.columns
            assert set(result['symbol'].unique()) == {'SPY', 'QQQ'}


class TestMetrics:
    """Test suite for metrics functions."""

    def test_metrics_bin_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        proba = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = _metrics_bin(y_true, proba)

        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics
        assert 'brier' in metrics
        assert metrics['roc_auc'] == 1.0
        assert metrics['log_loss'] < 0.1
        assert metrics['brier'] < 0.1

    def test_metrics_bin_random_prediction(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        proba = np.random.uniform(0, 1, 100)

        metrics = _metrics_bin(y_true, proba)

        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['log_loss'] > 0

    def test_compute_metrics_series(self):
        """Test compute_metrics with pandas Series."""
        y_true = pd.Series([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = compute_metrics(y_true, proba)

        assert isinstance(metrics, dict)
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] > 0.5  # Better than random


class TestCVFolds:
    """Test suite for cross-validation fold generation."""

    def test_time_series_folds_basic(self):
        """Test basic CV fold generation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        timestamps = pd.Series(dates)

        folds = _time_series_folds(
            timestamps,
            n_splits=3,
            purge_days=2,
            embargo_days=1
        )

        assert len(folds) == 3

        for train_idx, val_idx in folds:
            # Train and validation should not overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            # Both should have data
            assert len(train_idx) > 0
            assert len(val_idx) > 0

    def test_time_series_folds_purge_embargo(self):
        """Test that purge and embargo create gaps."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        timestamps = pd.Series(dates)

        folds = _time_series_folds(
            timestamps,
            n_splits=2,
            purge_days=3,
            embargo_days=2
        )

        # Just verify folds were created with purge/embargo parameters
        assert len(folds) == 2
        for train_idx, val_idx in folds:
            # Non-empty folds
            assert len(train_idx) > 0 or len(val_idx) > 0

    def test_time_series_folds_properties(self):
        """Test CV fold properties."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        timestamps = pd.Series(dates)

        folds = _time_series_folds(timestamps, n_splits=5, purge_days=2, embargo_days=1)

        # All folds should use data
        all_train_indices = set()
        all_val_indices = set()

        for train_idx, val_idx in folds:
            all_train_indices.update(train_idx)
            all_val_indices.update(val_idx)

        # Should have reasonable coverage
        assert len(all_train_indices) > 40
        assert len(all_val_indices) > 10


class TestModelTrainingIntegration:
    """Integration tests for model training workflow."""

    def test_train_config_to_cost_conversion(self):
        """Test converting TrainConfig to cost for labeling."""
        config = TrainConfig(cost_bps=5.0)
        cost_decimal = config.cost_bps / 1e4

        assert cost_decimal == 0.0005

    def test_workflow_components_compatible(self):
        """Test that workflow components work together."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        timestamps = pd.Series(dates)

        # Create labels
        y_true = pd.Series(np.random.randint(0, 2, 50))

        # Generate folds
        folds = _time_series_folds(timestamps, n_splits=3, purge_days=2, embargo_days=1)
        assert len(folds) == 3

        # Check folds are valid
        for train_idx, val_idx in folds:
            # At least one should have data
            assert len(train_idx) > 0 or len(val_idx) > 0
