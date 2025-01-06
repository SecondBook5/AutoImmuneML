import pytest
from src.data_loader import DataLoader
from src.utils.config_loader import ConfigLoader


@pytest.fixture
def real_config_loader():
    """
    Load the actual configuration using ConfigLoader.
    """
    config_path = "/home/secondbook5/projects/AutoImmuneML/config.yaml"  # Path to the actual config.yaml
    return ConfigLoader(config_path=config_path)


@pytest.fixture
def data_loader(real_config_loader):
    """
    Create an instance of DataLoader using the real configuration for crunch3.
    """
    return DataLoader(config=real_config_loader, crunch_name="crunch3")


def test_load_h5ad(data_loader):
    """
    Test the loading of .h5ad files.
    """
    result = data_loader.load_h5ad("scRNA_seq_file")
    assert result is not None  # Ensure something is returned


def test_load_tiff(data_loader):
    """
    Test the loading of .tiff files.
    """
    result = data_loader.load_tiff("he_image_file")
    assert result is not None  # Ensure something is returned


def test_load_zarr(data_loader):
    """
    Test the loading of .zarr files.
    """
    result = data_loader.load_zarr("raw_dir")
    assert result is not None  # Ensure something is returned


def test_load_csv(data_loader):
    """
    Test the loading of .csv files.
    """
    result = data_loader.load_csv("gene_list_file")
    assert result is not None  # Ensure something is returned


def test_load_all(data_loader):
    """
    Test loading all datasets for a Crunch.
    """
    result = data_loader.load_all()
    assert result  # Ensure the dictionary is not empty
