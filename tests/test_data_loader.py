import pytest
import pandas as pd
import numpy as np
from anndata import AnnData
from unittest.mock import patch
from src.data_loader import DataLoader
from src.loaders.h5ad_loader import H5ADLoader
from src.loaders.csv_loader import CSVLoader
from src.utils.config_loader import ConfigLoader


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


def test_validate_path(data_loader):
    """
    Test the path validation method.
    """
    path = data_loader.validate_path("scRNA_seq_file", is_file=True)
    assert path is not None  # Ensure a valid path is returned


@pytest.mark.parametrize("key, loader_class", [
    ("scRNA_seq_file", H5ADLoader),
    ("gene_list_file", CSVLoader),
])
def test_load_with_loader(data_loader, key, loader_class):
    """
    Test individual dataset loading using loaders.
    """
    with patch.object(loader_class, "load", return_value="mock_data"):
        result = data_loader._load_with_loader(key, loader_class)
        assert result == "mock_data"  # Ensure the mock data is returned


def test_load_batch(data_loader):
    """
    Test batch loading of datasets.
    """
    keys = ["scRNA_seq_file", "gene_list_file"]
    with patch("src.data_loader.DataLoader._load_with_loader", return_value="mock_data"):
        result = data_loader.load_batch(keys, H5ADLoader)
        assert len(result) == len(keys)  # Ensure all keys are processed


def test_load_zarr(data_loader):
    """
    Test the loading of Zarr files.
    """
    with patch("src.loaders.zarr_loader.ZARRLoader.load", return_value={"mock_key": "mock_value"}):
        result = data_loader.load_zarr(["raw_dir"])
        assert "raw_dir" in result  # Ensure the key exists in the result
        assert isinstance(result["raw_dir"], dict)  # Ensure the value is a dictionary


def test_stream_csv(data_loader):
    """
    Test streaming of .csv files in chunks.
    """
    mock_chunks = iter([pd.DataFrame({"mock": [1, 2]}), pd.DataFrame({"mock": [3, 4]})])
    with patch("pandas.read_csv", return_value=mock_chunks):
        generator = data_loader.stream_csv("gene_list_file", chunk_size=2)
        chunks = list(generator)
        assert len(chunks) > 0  # Ensure chunks are returned
        assert isinstance(chunks[0], pd.DataFrame)  # Ensure each chunk is a DataFrame



def test_stream_h5ad(data_loader):
    """
    Test streaming of .h5ad files in chunks.
    """
    mock_adata = AnnData(X=np.array([[1, 2], [3, 4], [5, 6]]))
    with patch("anndata.read_h5ad", return_value=mock_adata):
        generator = data_loader.stream_h5ad("scRNA_seq_file", chunk_size=1)
        chunks = list(generator)
        assert len(chunks) == 3  # Ensure correct number of chunks
        assert hasattr(chunks[0], "X")  # Ensure chunks have AnnData attributes


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
    with patch("pandas.read_csv", return_value=pd.DataFrame({"mock": [1, 2, 3]})):
        with patch("anndata.read_h5ad", return_value=mock_adata):
            result = data_loader.load_all(streaming=True)
            assert "h5ad_files_streams" in result
            assert "csv_files_streams" in result


def test_error_logging(data_loader, caplog):
    """
    Test logging for invalid paths.
    """
    # Attempt to validate a nonexistent path and capture the raised exception
    with pytest.raises(FileNotFoundError, match="Path Key 'nonexistent_file' not found in configuration for 'crunch3'"):
        data_loader.validate_path("nonexistent_file")

    # Check if the error message was logged
    assert any(
        "Path Key 'nonexistent_file' not found in configuration for 'crunch3'" in record.message
        for record in caplog.records
    ), "Expected error message not found in logs"




