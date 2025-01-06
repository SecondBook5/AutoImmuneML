# File: src/loaders/zarr_loader.py
import os  # For file and directory operations
import spatialdata as sd  # For working with SpatialData, a wrapper around Zarr datasets
from typing import Dict, Union  # For type annotations


class ZARRLoader:
    """
    Loader for Zarr files or directories containing multiple Zarr datasets.

    - Handles both single Zarr datasets and directories containing multiple `.zarr` groups.
    - Provides robust error handling and clear error messages for invalid paths or datasets.
    """

    def __init__(self, path: str):
        """
        Initialize the ZARRLoader with the path to a Zarr dataset or directory.

        Args:
            path (str): Path to a single Zarr file or a directory containing Zarr datasets.
        """
        self.path = path

    def load(self) -> Union[sd.SpatialData, Dict[str, sd.SpatialData]]:
        """
        Load Zarr data from the given path.

        - If the path points to a single `.zarr` dataset, return the SpatialData object.
        - If the path points to a directory, load all `.zarr` datasets in the directory.

        Returns:
            Union[sd.SpatialData, Dict[str, sd.SpatialData]]:
                - sd.SpatialData: If the path points to a single Zarr dataset.
                - Dict[str, sd.SpatialData]: A dictionary of SpatialData objects if the path points to a directory.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is neither a `.zarr` file nor a directory containing `.zarr` files.
        """
        # Ensure the specified path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        if os.path.isfile(self.path) and self.path.endswith(".zarr"):
            # Load a single Zarr dataset
            return self._load_single_zarr(self.path)
        elif os.path.isdir(self.path):
            # Load all Zarr datasets in the directory
            return self._load_all_zarr_in_directory()
        else:
            raise ValueError(f"Invalid Zarr path: {self.path}. Must be a .zarr file or a directory containing .zarr files.")

    def _load_single_zarr(self, zarr_path: str) -> sd.SpatialData:
        """
        Load a single Zarr dataset.

        Args:
            zarr_path (str): Path to a single Zarr dataset.

        Returns:
            sd.SpatialData: The loaded SpatialData object.

        Raises:
            ValueError: If the file is not a valid Zarr dataset.
        """
        try:
            return sd.read_zarr(zarr_path)
        except Exception as e:
            raise ValueError(f"Failed to load Zarr dataset at {zarr_path}: {e}")

    def _load_all_zarr_in_directory(self) -> Dict[str, sd.SpatialData]:
        """
        Load all `.zarr` datasets in the specified directory.

        Returns:
            Dict[str, sd.SpatialData]: A dictionary of loaded SpatialData objects.

        Raises:
            FileNotFoundError: If no `.zarr` files are found in the directory.
        """
        # Find all directories in the specified path that end with ".zarr"
        zarr_files = [
            os.path.join(self.path, item)
            for item in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, item)) and item.endswith(".zarr")
        ]
        # Check if any valid Zarr files were found
        if not zarr_files:
            raise FileNotFoundError(f"No valid .zarr datasets found in directory: {self.path}")

        # Load all Zarr datasets in the directory into a dictionary
        zarr_datasets = {}
        # Iterate over each Zarr file and load it
        for zarr_file in zarr_files:
            try:
                # Extract the dataset name from the file path (e.g., "UC1_NI.zarr")
                dataset_name = os.path.basename(zarr_file)
                # Load the Zarr dataset and store it in the dictionary with the dataset name
                zarr_datasets[dataset_name] = self._load_single_zarr(zarr_file)
            except Exception as e:
                print(f"Warning: Failed to load Zarr dataset {zarr_file}: {e}")
        # Return the dictionary of loaded Zarr datasets
        return zarr_datasets
