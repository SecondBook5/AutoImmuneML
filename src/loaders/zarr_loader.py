# File: src/loaders/zarr_loader.py
import os  # For file and directory operations
import zarr  # For working with Zarr datasets
from src.loaders.base_loader import BaseLoader  # Base class for loaders
from typing import Dict, Union  # For type annotations


class ZARRLoader(BaseLoader):
    """
    Loader for .zarr files or directories containing multiple .zarr datasets, inheriting from BaseLoader.

    - Handles both single Zarr datasets and directories containing multiple `.zarr` groups.
    - Provides robust error handling and logging for invalid paths or datasets.
    """

    def load(self) -> Union[zarr.Group, Dict[str, zarr.Group]]:
        """
        Load Zarr data.

        - If the path points to a single .zarr dataset, it loads and returns the Zarr group.
        - If the path points to a directory, it loads all valid .zarr datasets in the directory.

        Returns:
            Union[zarr.Group, Dict[str, zarr.Group]]:
                - zarr.Group: If the path points to a single Zarr dataset.
                - Dict[str, zarr.Group]: A dictionary of Zarr groups if the path points to a directory.

        Raises:
            FileNotFoundError: If the path does not exist.
            zarr.errors.GroupNotFoundError: If the path does not contain valid Zarr data.
        """
        # Ensure the specified path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        # Check if the path is a directory but not a single .zarr dataset
        if os.path.isdir(self.path) and not self.path.endswith(".zarr"):
            # Load all valid Zarr groups in the directory
            return self._load_all_zarr_groups()
        else:
            # Load a single Zarr group
            return self._load_single_group(self.path)

    def _load_single_group(self, path: str) -> zarr.Group:
        """
        Load a single Zarr group.

        Args:
            path (str): Path to a single Zarr group.

        Returns:
            zarr.Group: The loaded Zarr group.

        Raises:
            zarr.errors.GroupNotFoundError: If the group is not found.
        """
        try:
            # Attempt to open the specified path as a Zarr group
            return zarr.open_group(path, mode="r")
        except zarr.errors.GroupNotFoundError:
            # Raise a clear error if the path does not contain a valid Zarr group
            raise zarr.errors.GroupNotFoundError(f"Zarr group not found at path: {path}")

    def _load_all_zarr_groups(self) -> Dict[str, zarr.Group]:
        """
        Load all valid Zarr groups in the directory.

        Returns:
            Dict[str, zarr.Group]: A dictionary where keys are group names, and values are Zarr group objects.

        Raises:
            FileNotFoundError: If no valid `.zarr` datasets are found in the directory.
        """
        # Initialize an empty dictionary to store Zarr groups
        zarr_groups = {}

        # Iterate through each item in the directory
        for item in os.listdir(self.path):
            # Construct the full path for the current item
            item_path = os.path.join(self.path, item)

            # Check if the item is a directory and ends with '.zarr'
            if os.path.isdir(item_path) and item.endswith(".zarr"):
                try:
                    # Attempt to load the Zarr group and add it to the dictionary
                    zarr_groups[item] = self._load_single_group(item_path)
                except zarr.errors.GroupNotFoundError:
                    # Log a warning and skip the item if it is not a valid Zarr group
                    print(f"Warning: {item_path} is not a valid Zarr group. Skipping...")

        # If no valid Zarr groups were found, raise an error
        if not zarr_groups:
            raise FileNotFoundError(f"No valid Zarr datasets found in directory: {self.path}")

        # Return the dictionary of loaded Zarr groups
        return zarr_groups
