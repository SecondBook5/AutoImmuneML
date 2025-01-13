# File: src/loaders/zarr_loader.py
import os  # For file and directory operations
import spatialdata as sd  # For working with SpatialData, a wrapper around Zarr datasets
from typing import Dict, Union  # For type annotations
from src.loaders.base_loader import BaseLoader  # Import BaseLoader


class ZARRLoader(BaseLoader):
    """
    Loader for Zarr files or directories containing multiple Zarr datasets.

    - Extends BaseLoader for path validation.
    - Handles both single Zarr datasets and directories containing multiple `.zarr` groups.
    - Provides robust error handling for invalid paths or datasets.
    """

    def load(self) -> Union[sd.SpatialData, Dict[str, sd.SpatialData]]:
        """
        Load Zarr data from the given path.

        - If the path points to a single `.zarr` dataset, load it.
        - If the path points to a directory, load all `.zarr` datasets in the directory.

        Returns:
            Union[sd.SpatialData, Dict[str, sd.SpatialData]]:
                - sd.SpatialData: If the path points to a single Zarr dataset.
                - Dict[str, sd.SpatialData]: A dictionary of SpatialData objects if the path points to a directory.

        Raises:
            ValueError: If the path is neither a `.zarr` directory nor a directory containing `.zarr` datasets.
        """
        # Determine whether the path points to a single dataset or a directory
        if os.path.isdir(self.path) and self.path.endswith(".zarr"):
            # Load a single Zarr dataset
            return self._load_single_zarr(self.path)
        elif os.path.isdir(self.path):
            # Load all Zarr datasets in the directory
            return self._load_all_zarr_in_directory()
        else:
            raise ValueError(
                f"Invalid Zarr path: {self.path}. Must be a .zarr directory or a directory containing .zarr datasets."
            )

    def _load_single_zarr(self, zarr_path: str) -> sd.SpatialData:
        """
        Load a single Zarr dataset.

        Args:
            zarr_path (str): Path to a single Zarr dataset.

        Returns:
            sd.SpatialData: The loaded SpatialData object.

        Raises:
            ValueError: If the `.zarr` directory structure is invalid or unreadable.
        """
        try:
            # Load the Zarr dataset using SpatialData
            return sd.read_zarr(zarr_path)
        except Exception as e:
            # Raise an error if the dataset cannot be loaded
            raise ValueError(f"Failed to load Zarr dataset at {zarr_path}: {e}")

    def _load_all_zarr_in_directory(self) -> Dict[str, sd.SpatialData]:
        """
        Load all `.zarr` datasets in the specified directory.

        Returns:
            Dict[str, sd.SpatialData]: A dictionary of loaded SpatialData objects.

        Raises:
            FileNotFoundError: If no `.zarr` datasets are found in the directory.
        """
        # Find all `.zarr` directories in the specified path
        zarr_files = [
            os.path.join(self.path, item)
            for item in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, item)) and item.endswith(".zarr")
        ]

        # Check if any `.zarr` datasets were found
        if not zarr_files:
            raise FileNotFoundError(f"No valid .zarr datasets found in directory: {self.path}")

        # Initialize a dictionary to store loaded datasets
        zarr_datasets = {}
        # Iterate over each `.zarr` directory in the path
        for zarr_file in zarr_files:
            try:
                # Extract the dataset name from the file path (e.g., "UC1_NI.zarr")
                dataset_name = os.path.basename(zarr_file)
                # Load the Zarr dataset
                zarr_datasets[dataset_name] = self._load_single_zarr(zarr_file)
            except Exception as e:
                # Print a warning if a dataset fails to load, but continue processing others
                print(f"Warning: Failed to load Zarr dataset {zarr_file}: {e}")

        # Return the dictionary of loaded datasets
        return zarr_datasets
