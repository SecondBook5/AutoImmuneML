import pytest
import pandas as pd
import anndata as ad
from src.data_loader import DataLoader
from src.utils.config_loader import ConfigLoader
from unittest.mock import patch


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


def test_validate_path(data_loader):
    """
    Test the path validation method.
    """
    path = data_loader.validate_path("scRNA_seq_file", is_file=True)
    assert path is not None  # Ensure a valid path is returned


@pytest.mark.parametrize("key, loader_class", [
    ("scRNA_seq_file", ad.read_h5ad),  # Mock loading for .h5ad
    ("gene_list_file", pd.read_csv),  # Mock loading for .csv
])
def test_load_with_loader(data_loader, key, loader_class):
    """
    Test the _load_with_loader method for multiple dataset types.
    """
    with patch(f"{loader_class.__module__}.{loader_class.__name__}") as mock_loader:
        mock_loader.return_value = "mock_data"
        result = data_loader._load_with_loader(key, loader_class)
        assert result == "mock_data"  # Ensure mocked data is returned


def test_load_batch(data_loader):
    """
    Test batch loading of datasets.
    """
    keys = ["scRNA_seq_file", "gene_list_file"]
    with patch("src.data_loader.DataLoader._load_with_loader", return_value="mock_data"):
        result = data_loader.load_batch(keys, lambda x: x)
        assert result  # Ensure batch loading returns results
        assert len(result) == len(keys)  # Ensure all keys are loaded


def test_load_zarr(data_loader):
    """
    Test the loading of Zarr files or directories.
    """
    with patch("src.loaders.zarr_loader.ZARRLoader.load", return_value={"mock_key": "mock_zarr"}):
        result = data_loader.load_zarr(["raw_dir"])
        assert isinstance(result, dict)  # Ensure a dictionary is returned
        assert "raw_dir" in result  # Ensure keys match input


def test_stream_csv(data_loader):
    """
    Test streaming of .csv files in chunks.
    """
    with patch("pandas.read_csv", return_value=pd.DataFrame({"mock": [1, 2, 3]})):
        generator = data_loader.stream_csv("gene_list_file", chunk_size=2)
        chunks = list(generator)
        assert len(chunks) > 0  # Ensure chunks are generated
        assert isinstance(chunks[0], pd.DataFrame)  # Ensure chunks are DataFrames


def test_stream_h5ad(data_loader):
    """
    Test streaming of .h5ad files in chunks.
    """
    mock_adata = ad.AnnData(X=[[1, 2], [3, 4], [5, 6]])
    with patch("anndata.read_h5ad", return_value=mock_adata):
        generator = data_loader.stream_h5ad("scRNA_seq_file", chunk_size=1)
        chunks = list(generator)
        assert len(chunks) == 3  # Ensure correct number of chunks
        assert hasattr(chunks[0], "X")  # Ensure chunks have AnnData attributes


def test_load_all_non_streaming(data_loader):
    """
    Test loading all datasets without streaming.
    """
    with patch("src.data_loader.DataLoader.load_batch", return_value={"mock": "mock_data"}):
        with patch("src.data_loader.DataLoader.load_zarr", return_value={"mock": "mock_zarr"}):
            result = data_loader.load_all(streaming=False)
            assert "h5ad_files" in result
            assert "zarr_files" in result
            assert "csv_files" in result
            assert "tiff_files" in result


def test_load_all_streaming(data_loader):
    """
    Test loading all datasets with streaming enabled.
    """
    with patch("pandas.read_csv", return_value=pd.DataFrame({"mock": [1, 2, 3]})):
        with patch("anndata.read_h5ad", return_value=ad.AnnData(X=[[1, 2], [3, 4], [5, 6]])):
            result = data_loader.load_all(streaming=True)
            assert "h5ad_files_streams" in result
            assert "csv_files_streams" in result
            for stream in result["h5ad_files_streams"].values():
                assert isinstance(list(stream), list)


def test_error_logging_on_failure(data_loader, caplog):
    """
    Test logging for missing or invalid paths during dataset loading.
    """
    with pytest.raises(FileNotFoundError):
        data_loader.validate_path("nonexistent_file")
    assert "Path not found" in caplog.text  # Ensure the log contains the error
