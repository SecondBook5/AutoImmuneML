# File: tests/test_config_loader.py
import pytest
from src.config.config_loader import ConfigLoader


@pytest.fixture
def sample_config(tmp_path):
    """
    Creates a temporary sample configuration file for testing.
    """
    config_content = """
    global:
      token_file: "/path/to/token"
      base_dir: "/base/dir"
      src_dir: "/src/dir"
      data_dir: "/mnt/d/AutoImmuneML"

    crunches:
      crunch1:
        name: "autoimmune-crunch1"
        paths:
          project_dir: "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1"
          raw_dir: "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1/raw"
        training:
          batch_size: 128
          seed: 42
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def config_loader(sample_config):
    """
    Returns an instance of ConfigLoader initialized with the sample config.
    """
    return ConfigLoader(config_path=sample_config)


def test_load_config_valid(sample_config):
    """
    Test loading a valid configuration file.
    """
    loader = ConfigLoader(config_path=sample_config)
    assert loader.config is not None
    assert isinstance(loader.config, dict)


def test_load_config_invalid(tmp_path):
    """
    Test loading an invalid configuration file.
    """
    invalid_config_file = tmp_path / "invalid_config.yaml"
    invalid_config_file.write_text("invalid_yaml: [")
    with pytest.raises(ValueError):
        ConfigLoader(config_path=str(invalid_config_file))


def test_load_config_file_not_found():
    """
    Test loading a configuration file that does not exist.
    """
    with pytest.raises(FileNotFoundError):
        ConfigLoader(config_path="/nonexistent/config.yaml")


def test_get_global_setting(config_loader):
    """
    Test retrieving a global setting from the configuration.
    """
    assert config_loader.get_global_setting("token_file") == "/path/to/token"
    assert config_loader.get_global_setting("nonexistent_key", "default_value") == "default_value"


def test_get_crunch_setting(config_loader):
    """
    Test retrieving a specific Crunch setting.
    """
    assert config_loader.get_crunch_setting("crunch1", "name") == "autoimmune-crunch1"
    assert config_loader.get_crunch_setting("crunch1", "nonexistent_key", "default_value") == "default_value"
    assert config_loader.get_crunch_setting("nonexistent_crunch", "name", "default_value") == "default_value"


def test_get_crunch_path(config_loader):
    """
    Test retrieving a path for a specific Crunch.
    """
    assert config_loader.get_crunch_path("crunch1", "project_dir") == "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1"
    assert config_loader.get_crunch_path("crunch1", "nonexistent_path", "/default/path") == "/default/path"
    assert config_loader.get_crunch_path("nonexistent_crunch", "project_dir", "/default/path") == "/default/path"


def test_get_training_setting(config_loader):
    """
    Test retrieving a training-related setting for a specific Crunch.
    """
    assert config_loader.get_training_setting("crunch1", "batch_size") == 128
    assert config_loader.get_training_setting("crunch1", "nonexistent_key", "default_value") == "default_value"
    assert config_loader.get_training_setting("nonexistent_crunch", "batch_size", "default_value") == "default_value"


def test_get_preprocessing_setting(config_loader):
    """
    Test retrieving a preprocessing-related setting for a specific Crunch.
    """
    assert config_loader.get_preprocessing_setting("crunch1", "nonexistent_key", "default_value") == "default_value"
    assert config_loader.get_preprocessing_setting("nonexistent_crunch", "sample_size", 1000) == 1000
