"""
Unit tests for quantfund configuration system.
"""
import tempfile
import yaml
from pathlib import Path
import pytest

from quantfund.models.train import TrainConfig, load_train_config, _read_yaml_config


class TestTrainConfig:
    """Test suite for training configuration."""
    
    def test_default_train_config(self):
        """Test default training configuration values."""
        config = TrainConfig()
        
        assert config.cost_bps == 2.0
        assert config.cv_splits == 5
        assert config.purge_days == 5
        assert config.embargo_days == 2
    
    def test_custom_train_config(self):
        """Test custom training configuration values."""
        config = TrainConfig(
            cost_bps=3.5,
            cv_splits=3,
            purge_days=7,
            embargo_days=1
        )
        
        assert config.cost_bps == 3.5
        assert config.cv_splits == 3
        assert config.purge_days == 7
        assert config.embargo_days == 1
    
    def test_train_config_immutability(self):
        """Test that training config is immutable (frozen dataclass)."""
        config = TrainConfig()
        
        with pytest.raises(AttributeError):
            config.cost_bps = 5.0


class TestYamlConfigReader:
    """Test suite for YAML configuration reader."""
    
    def test_read_existing_yaml_config(self):
        """Test reading an existing YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                'cost_bps': 2.5,
                'cv_splits': 4,
                'purge_days': 3,
                'embargo_days': 1
            }
            yaml.dump(test_config, f)
            temp_path = Path(f.name)
        
        try:
            config = _read_yaml_config(temp_path)
            
            assert config['cost_bps'] == 2.5
            assert config['cv_splits'] == 4
            assert config['purge_days'] == 3
            assert config['embargo_days'] == 1
        finally:
            temp_path.unlink()  # Clean up
    
    def test_read_nonexistent_yaml_config(self):
        """Test reading a non-existent YAML configuration file."""
        nonexistent_path = Path("nonexistent_config.yaml")
        
        config = _read_yaml_config(nonexistent_path)
        
        assert config == {}
    
    def test_read_empty_yaml_config(self):
        """Test reading an empty YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = Path(f.name)
        
        try:
            config = _read_yaml_config(temp_path)
            assert config == {}
        finally:
            temp_path.unlink()
    
    def test_read_invalid_yaml_config(self):
        """Test reading an invalid YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(yaml.YAMLError):
                _read_yaml_config(temp_path)
        finally:
            temp_path.unlink()


class TestLoadTrainConfig:
    """Test suite for loading training configuration from file."""
    
    def test_load_train_config_with_existing_file(self):
        """Test loading training config when file exists."""
        # Create a temporary config file in the expected location
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        
        train_yaml_path = configs_dir / "train.yaml"
        
        # Save original content if it exists
        original_content = None
        if train_yaml_path.exists():
            with open(train_yaml_path, 'r') as f:
                original_content = f.read()
        
        try:
            # Write test configuration
            test_config = {
                'cost_bps': 4.0,
                'cv_splits': 3,
                'purge_days': 7,
                'embargo_days': 3
            }
            
            with open(train_yaml_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Load configuration
            config = load_train_config()
            
            assert config.cost_bps == 4.0
            assert config.cv_splits == 3
            assert config.purge_days == 7
            assert config.embargo_days == 3
            
        finally:
            # Restore original content or remove test file
            if original_content is not None:
                with open(train_yaml_path, 'w') as f:
                    f.write(original_content)
            elif train_yaml_path.exists():
                train_yaml_path.unlink()
    
    def test_load_train_config_with_missing_file(self):
        """Test loading training config when file doesn't exist."""
        # Temporarily rename the config file if it exists
        configs_dir = Path("configs")
        train_yaml_path = configs_dir / "train.yaml"
        backup_path = configs_dir / "train.yaml.backup"
        
        file_existed = train_yaml_path.exists()
        if file_existed:
            train_yaml_path.rename(backup_path)
        
        try:
            # Load configuration (should use defaults)
            config = load_train_config()
            
            # Should use default values
            assert config.cost_bps == 2.0
            assert config.cv_splits == 5
            assert config.purge_days == 5
            assert config.embargo_days == 2
            
        finally:
            # Restore original file
            if file_existed and backup_path.exists():
                backup_path.rename(train_yaml_path)
    
    def test_load_train_config_with_partial_config(self):
        """Test loading training config with only some values specified."""
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        
        train_yaml_path = configs_dir / "train.yaml"
        
        # Save original content if it exists
        original_content = None
        if train_yaml_path.exists():
            with open(train_yaml_path, 'r') as f:
                original_content = f.read()
        
        try:
            # Write partial configuration (only some values)
            partial_config = {
                'cost_bps': 3.5,
                'cv_splits': 7
                # Missing purge_days and embargo_days
            }
            
            with open(train_yaml_path, 'w') as f:
                yaml.dump(partial_config, f)
            
            # Load configuration
            config = load_train_config()
            
            # Should use specified values and defaults for missing ones
            assert config.cost_bps == 3.5
            assert config.cv_splits == 7
            assert config.purge_days == 5  # Default
            assert config.embargo_days == 2  # Default
            
        finally:
            # Restore original content or remove test file
            if original_content is not None:
                with open(train_yaml_path, 'w') as f:
                    f.write(original_content)
            elif train_yaml_path.exists():
                train_yaml_path.unlink()
    
    def test_load_train_config_type_conversion(self):
        """Test that configuration values are properly type-converted."""
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        
        train_yaml_path = configs_dir / "train.yaml"
        
        # Save original content if it exists
        original_content = None
        if train_yaml_path.exists():
            with open(train_yaml_path, 'r') as f:
                original_content = f.read()
        
        try:
            # Write configuration with string values that should be converted
            string_config = {
                'cost_bps': '2.5',  # String that should become float
                'cv_splits': '6',   # String that should become int
                'purge_days': '4',  # String that should become int
                'embargo_days': '1' # String that should become int
            }
            
            with open(train_yaml_path, 'w') as f:
                yaml.dump(string_config, f)
            
            # Load configuration
            config = load_train_config()
            
            # Should properly convert types
            assert config.cost_bps == 2.5
            assert isinstance(config.cost_bps, float)
            assert config.cv_splits == 6
            assert isinstance(config.cv_splits, int)
            assert config.purge_days == 4
            assert isinstance(config.purge_days, int)
            assert config.embargo_days == 1
            assert isinstance(config.embargo_days, int)
            
        finally:
            # Restore original content or remove test file
            if original_content is not None:
                with open(train_yaml_path, 'w') as f:
                    f.write(original_content)
            elif train_yaml_path.exists():
                train_yaml_path.unlink()


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_consistency_across_modules(self):
        """Test that configuration is consistent across different modules."""
        from quantfund.models.train import get_cost_from_config
        
        # Load config and get cost
        config = load_train_config()
        cost_decimal = get_cost_from_config()
        
        # Cost should be properly converted from bps to decimal
        expected_cost = config.cost_bps / 1e4
        assert abs(cost_decimal - expected_cost) < 1e-10
        
        # Cost should be reasonable (between 0.1 and 50 bps)
        assert 1e-5 <= cost_decimal <= 5e-3
    
    def test_config_validation_bounds(self):
        """Test that configuration values are within reasonable bounds."""
        config = load_train_config()
        
        # Cost should be reasonable (0.5 to 10 bps)
        assert 0.5 <= config.cost_bps <= 10.0
        
        # CV splits should be reasonable (2 to 10)
        assert 2 <= config.cv_splits <= 10
        
        # Purge days should be reasonable (1 to 30)
        assert 1 <= config.purge_days <= 30
        
        # Embargo days should be reasonable (0 to 10)
        assert 0 <= config.embargo_days <= 10