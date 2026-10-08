"""
Pytest configuration and shared fixtures for quantfund tests.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    
    # Generate realistic price series
    returns = np.random.normal(0.0001, 0.01, 100)
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC with realistic relationships
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.002, 100)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.002, 100)))
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)


@pytest.fixture
def sample_features_data(sample_ohlc_data):
    """Generate sample feature data based on OHLC prices."""
    df = sample_ohlc_data.copy()
    
    # Add technical indicators
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['zscore_20'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()
    df['rsi_14'] = 50 + 50 * np.tanh(df['returns'].rolling(14).mean() / df['returns'].rolling(14).std())
    
    # Forward-looking columns for labeling
    df['next_open'] = df['open'].shift(-1)
    df['next_close'] = df['close'].shift(-1)
    
    return df


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a mock configuration file for testing."""
    config_content = """
cost_bps: 2.5
cv_splits: 4
purge_days: 3
embargo_days: 1
reports_dir: reports
intervals:
  - '60m'
  - '120m'
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture(autouse=True)
def ensure_reports_dir():
    """Ensure reports directory exists for tests."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    yield reports_dir


@pytest.fixture
def clean_test_artifacts():
    """Clean up test artifacts after each test."""
    yield
    
    # Clean up any test files that might have been created
    test_patterns = [
        "test_*.parquet",
        "test_*.json",
        "test_*.png",
        "*_test.*"
    ]
    
    for pattern in test_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                file_path.unlink()
    
    # Clean up test directories
    test_dirs = ["test_data", "test_reports", "test_artifacts"]
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists() and test_path.is_dir():
            shutil.rmtree(test_path)


# Configure numpy and pandas for testing
@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure test environment settings."""
    # Set numpy random seed for reproducible tests
    np.random.seed(42)
    
    # Configure pandas display options for testing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    yield
    
    # Reset pandas options after tests
    pd.reset_option('all')


# Custom markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data"
    )


# Skip slow tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests."""
    if config.getoption("--run-slow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true", 
        default=False,
        help="run integration tests"
    )