# File: tests/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
from unittest.mock import patch
from src.data_loader import DataLoader
from src.loaders.h5ad_loader import H5ADLoader
from src.config.config_loader import ConfigLoader
from src.validators.zarr_validator import ZARRValidator


@pytest.fixture
def real_config_loader():
    """
    Load the actual configuration using ConfigLoader.
    """
    config_path = "/home/secondbook5/projects/AutoImmuneML/config.yaml"
    return ConfigLoader(config_path=config_path)


@pytest.fixture
def data_loader(real_config_loader):
    """
    Create an instance of DataLoader using the real configuration for crunch3.
    """
    return DataLoader(config=real_config_loader, crunch_name="crunch3")


def test_validate_zarr(data_loader):
    """
    Test the Zarr validation method.
    """
    valid_zarr_path = "/mnt/d/AutoImmuneML/broad-3-autoimmune-crunch3/data/valid.zarr"
    invalid_zarr_path = "/mnt/d/AutoImmuneML/broad-3-autoimmune-crunch3/data/invalid.zarr"

    with patch.object(ZARRValidator, "validate", side_effect=[
        {"status": "valid", "errors": []},
        {"status": "invalid", "errors": ["Invalid structure"]}
    ]):
        assert data_loader.validate_zarr(valid_zarr_path) is True
        assert data_loader.validate_zarr(invalid_zarr_path) is False


def test_load_zarr(data_loader):
    """
    Test the loading of Zarr files and handling of directories containing multiple Zarr datasets.
    """
    valid_zarr_paths = [
        "/mnt/d/AutoImmuneML/broad-3-autoimmune-crunch3/data/valid1.zarr",
        "/mnt/d/AutoImmuneML/broad-3-autoimmune-crunch3/data/valid2.zarr"
    ]

    # Mock os.path.exists to simulate valid paths
    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", side_effect=lambda path: path.endswith(".zarr")), \
         patch.object(ZARRValidator, "validate", return_value={"status": "valid", "errors": []}), \
         patch("src.loaders.zarr_loader.ZARRLoader.load", return_value="mock_data"):
        result = data_loader.load_zarr(valid_zarr_paths)
        assert len(result) == len(valid_zarr_paths)  # Ensure all datasets are loaded
        assert all(value == "mock_data" for value in result.values())

def test_load_batch(data_loader):
    """
    Test batch loading of datasets.
    """
    keys = ["scRNA_seq_file", "gene_list_file"]

    # Mock the loader class's `load` method directly and skip path validation
    with patch("src.loaders.h5ad_loader.H5ADLoader.__init__", return_value=None), \
         patch("src.loaders.h5ad_loader.H5ADLoader.load", return_value="mock_data"):
        # Re-initialize the loader to skip path validation
        result = data_loader.load_batch(keys, H5ADLoader)
        assert len(result) == len(keys)  # Ensure all keys are processed
        assert all(value == "mock_data" for value in result.values())  # Ensure all results match the mocked data


def test_stream_csv(data_loader):
    """
    Test streaming of .csv files in chunks.
    """
    mock_chunks = iter([pd.DataFrame({"mock": [1, 2]}), pd.DataFrame({"mock": [3, 4]})])
    with patch("os.path.isfile", return_value=True), patch("pandas.read_csv", return_value=mock_chunks):
        generator = data_loader.stream_csv("gene_list_file", chunk_size=2)
        chunks = list(generator)
        assert len(chunks) > 0
        assert isinstance(chunks[0], pd.DataFrame)


def test_stream_h5ad(data_loader):
    """
    Test streaming of .h5ad files in chunks.
    """
    mock_adata = AnnData(X=np.array([[1, 2], [3, 4], [5, 6]]))
    with patch("os.path.isfile", return_value=True), patch("anndata.read_h5ad", return_value=mock_adata):
        generator = data_loader.stream_h5ad("scRNA_seq_file", chunk_size=1)
        chunks = list(generator)
        assert len(chunks) == 3
        assert hasattr(chunks[0], "X")


def test_load_all(data_loader):
    """
    Test loading all datasets without streaming.
    """
    with patch("src.data_loader.DataLoader.load_batch", return_value={"mock_key": "mock_value"}):
        result = data_loader.load_all()
        assert "h5ad_files" in result
        assert "zarr_files" in result


def test_load_all_streaming(data_loader):
    """
    Test loading all datasets with streaming enabled.
    """
    mock_adata = AnnData(X=np.array([[1, 2], [3, 4], [5, 6]]))
    with patch("pandas.read_csv", return_value=pd.DataFrame({"mock": [1, 2, 3]})), \
         patch("anndata.read_h5ad", return_value=mock_adata), \
         patch("os.path.exists", return_value=True), \
         patch("src.loaders.tiff_loader.TIFFLoader.load", return_value="mock_data"):
        result = data_loader.load_all(streaming=True)
        assert "h5ad_files_streams" in result
        assert "csv_files_streams" in result
