import pytest
import pandas as pd
import numpy as np
import os
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
    Test the loading of Zarr files and handling of directories containing multiple Zarr datasets.
    """
    # Ensure the configuration contains a valid path for "raw_dir"
    zarr_key = "raw_dir"
    zarr_path = data_loader.config.get_crunch_path("crunch3", zarr_key)
    assert zarr_path, f"Configuration missing path for key '{zarr_key}' in 'crunch3'."

    # Check that the path exists and is a directory
    assert os.path.exists(zarr_path), f"Path '{zarr_path}' does not exist."
    assert os.path.isdir(zarr_path), f"Path '{zarr_path}' is not a directory."

    # Load the Zarr dataset
    result = data_loader.load_zarr([zarr_key])

    # Assertions to verify the loaded data
    expected_key = "UC9_I.zarr"  # Adjust this key based on the actual Zarr dataset name
    assert expected_key in result, f"Expected Zarr dataset '{expected_key}' not found in result."
    loaded_data = result[expected_key]

    # Check that the loaded data is a SpatialData object
    from spatialdata import SpatialData
    assert isinstance(loaded_data, SpatialData), f"Loaded data for '{expected_key}' is not a SpatialData object."

    # Verify the presence of expected components in the SpatialData object
    assert "HE_nuc_original" in loaded_data.images.keys(), "'HE_nuc_original' not found in images."
    assert "HE_original" in loaded_data.images.keys(), "'HE_original' not found in images."
    assert "anucleus" in loaded_data.tables.keys(), "'anucleus' not found in tables."
    assert "cell_id-group" in loaded_data.tables.keys(), "'cell_id-group' not found in tables."





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




