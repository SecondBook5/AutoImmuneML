# File: tests/test_spatialdata_handler.py
import pytest
import os
import numpy as np
import pandas as pd
from spatialdata import SpatialData  # Mocked spatialdata for testing
from src.pipelines.spatialdata_handler import SpatialDataHandler
from unittest.mock import MagicMock, patch

# Fixtures for setup and teardown
@pytest.fixture
def mock_zarr_paths(tmp_path):
    """Fixture to create temporary Zarr paths for testing."""
    zarr_paths = []
    for i in range(3):
        zarr_path = tmp_path / f"dataset_{i}.zarr"
        zarr_path.mkdir()
        zarr_paths.append(str(zarr_path))
    return zarr_paths

@pytest.fixture
def handler(mock_zarr_paths):
    """Fixture to initialize a fresh SpatialDataHandler for each test."""
    return SpatialDataHandler(mock_zarr_paths)

# Mock helper for creating fake SpatialData objects
def create_mock_spatialdata():
    mock_data = MagicMock(spec=SpatialData)
    mock_data.tables = {
        "anucleus": MagicMock(
            obs_names=np.array([f"nucleus_{i}" for i in range(100)]),
            var_names=np.array([f"gene_{j}" for j in range(10)]),
            X=np.random.rand(100, 10),
            obsm={"spatial": np.random.rand(100, 2)},
        )
    }
    mock_data.images = {"HE_original": MagicMock(to_numpy=lambda: np.random.rand(3, 256, 256))}
    return mock_data

# Test cases for `load_data`
def test_load_data_success(handler, mock_zarr_paths):
    """Test successful data loading."""
    with patch("spatialdata.read_zarr", side_effect=lambda x: create_mock_spatialdata()):
        handler.load_data()
        assert len(handler.datasets) == len(mock_zarr_paths)
        assert all(dataset in handler.datasets for dataset in map(os.path.basename, mock_zarr_paths))

def test_load_data_failure(handler, mock_zarr_paths):
    """Test failure to load data with retries."""
    def mock_failure(path):
        if "dataset_1" in path:
            raise ValueError("Mock loading error")
        return create_mock_spatialdata()

    with patch("spatialdata.read_zarr", side_effect=mock_failure):
        handler.load_data(max_retries=2)
        assert "dataset_1.zarr" not in handler.datasets

# Test cases for `validate_data`
def test_validate_data_success(handler):
    """Test successful validation of datasets."""
    with patch("spatialdata.read_zarr", side_effect=lambda x: create_mock_spatialdata()):
        handler.load_data()
    handler.validate_data(required_images=["HE_original"], required_tables=["anucleus"])

def test_validate_data_missing_component(handler):
    """Test validation failure due to missing components."""
    mock_data = create_mock_spatialdata()

    # Simulate missing components
    mock_data.images.pop("HE_nuc_original", None)
    mock_data.tables.pop("anucleus", None)

    # Patch `read_zarr` to return the mock dataset
    with patch("spatialdata.read_zarr", side_effect=lambda x: mock_data):
        handler.load_data()
        with pytest.raises(ValueError, match="Dataset .* is missing the following components"):
            handler.validate_data()

# Test cases for `subsample_data`
def test_subsample_data(handler):
    """Test successful subsampling of datasets."""
    mock_data = create_mock_spatialdata()

    # Mock table with proper slicing behavior
    mock_table = MagicMock()
    mock_table.shape = (1000, 460)
    mock_table.obs_names = np.array([f"cell_{i}" for i in range(1000)])

    # Define slicing behavior
    def mock_getitem(key):
        subsample_size = len(key) if isinstance(key, (list, np.ndarray)) else 10
        mock_table.shape = (subsample_size, 460)
        return mock_table

    mock_table.__getitem__.side_effect = mock_getitem
    mock_data.tables["anucleus"] = mock_table

    # Patch `read_zarr` to use the mock dataset
    with patch("spatialdata.read_zarr", side_effect=lambda x: mock_data):
        handler.load_data()
        handler.subsample_data(max_cells=10)

        for dataset in handler.datasets.values():
            assert dataset.tables["anucleus"].shape[0] == 10

# Test cases for `extract_nuclei_and_gene_expression`
def test_extract_nuclei_and_gene_expression(handler):
    """Test extraction of nuclei and gene expression."""
    with patch("spatialdata.read_zarr", side_effect=lambda x: create_mock_spatialdata()):
        handler.load_data()
    results = handler.extract_nuclei_and_gene_expression(batch_size=5)
    for dataset_name, df in results.items():
        assert isinstance(df, pd.DataFrame)
        assert "nuclei_id" in df.columns
        assert "x_coord" in df.columns
        assert "y_coord" in df.columns

# Test cases for caching and image retrieval
def test_get_image(handler):
    """Test lazy loading and retrieval of images."""
    with patch("spatialdata.read_zarr", side_effect=lambda x: create_mock_spatialdata()):
        handler.load_data()
    image = handler.get_image("dataset_0.zarr", "HE_original")
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, 256, 256)

# Test interactive visualization
def test_interactive_visualize(handler):
    """Test interactive visualization creation."""
    mock_data = create_mock_spatialdata()
    mock_image = np.random.rand(256, 256, 3)  # Generate a mock RGB image
    mock_data.images["HE_original"].to_numpy = lambda: mock_image  # Return the mock image
    with patch("spatialdata.read_zarr", side_effect=lambda x: mock_data):
        handler.load_data()
        handler.interactive_visualize("dataset_0.zarr", "HE_original")  # Should not raise errors

# Test for `print_summary`
def test_print_summary(handler, capsys):
    """Test detailed summary output."""
    with patch("spatialdata.read_zarr", side_effect=lambda x: create_mock_spatialdata()):
        handler.load_data()
    handler.print_summary()
    captured = capsys.readouterr()
    assert "Dataset" in captured.out
    assert "Images" in captured.out
    assert "Tables" in captured.out
