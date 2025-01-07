# File: src/downloader/manifest_manager.py

import os
import json
from typing import Dict, List


class ManifestManager:
    """
    A utility class to handle the manifest file for tracking downloads.

    This class provides methods to load, save, update, query, and dynamically generate the manifest.
    """

    def __init__(self, manifest_path: str):
        """
        Initialize the ManifestManager with the path to the manifest file.

        Args:
            manifest_path (str): Path to the manifest file.
        """
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """
        Load the manifest file into memory.

        Returns:
            Dict: The loaded manifest data.

        Raises:
            ValueError: If the manifest file has invalid JSON syntax.
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing manifest JSON: {e}")

        # Return an empty dictionary if the manifest does not exist
        return {}

    def save_manifest(self) -> None:
        """
        Save the manifest data to the manifest file.
        """
        try:
            with open(self.manifest_path, "w") as file:
                json.dump(self.manifest, file, indent=4)
            print(f"[INFO] Manifest successfully saved to {self.manifest_path}.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Unexpected error saving manifest: {e}")

    def is_downloaded(self, crunch_name: str, file_path: str) -> bool:
        """
        Check if a file or directory is already downloaded.

        Args:
            crunch_name (str): Name of the Crunch.
            file_path (str): Path of the file to check.

        Returns:
            bool: True if the file is downloaded, False otherwise.
        """
        return self.manifest.get(crunch_name, {}).get(file_path, {}).get("status") == "downloaded"

    def update_manifest(self, crunch_name: str, file_path: str, status: str, size: int = 0) -> None:
        """
        Update the manifest with the status of a file.

        Args:
            crunch_name (str): Name of the Crunch.
            file_path (str): Path of the file.
            status (str): Status of the file (e.g., "downloaded").
            size (int): Size of the file in bytes.
        """
        if crunch_name not in self.manifest:
            self.manifest[crunch_name] = {}

        self.manifest[crunch_name][file_path] = {
            "status": status,
            "size_bytes": size
        }

        # Save the manifest immediately after update
        self.save_manifest()

    def calculate_directory_size(self, directory: str) -> int:
        """
        Calculate the total size of all files in a directory, including subdirectories.

        Args:
            directory (str): The directory to calculate size for.

        Returns:
            int: Total size in bytes.
        """
        total_size = 0
        for root, _, files in os.walk(directory):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size

    def generate_manifest_entry(self, directory: str, file_types: List[str]) -> Dict:
        """
        Generate a manifest entry for a directory, focusing on specified file types.

        Args:
            directory (str): Directory to scan for files.
            file_types (List[str]): List of file extensions to include (e.g., ".tif", ".zarr").

        Returns:
            Dict: A dictionary representing the manifest entry.
        """
        manifest_entry = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in file_types):
                    relative_path = os.path.relpath(os.path.join(root, file), directory)
                    file_size = os.path.getsize(os.path.join(root, file))
                    manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": file_size}

            for dir_name in dirs:
                if dir_name.endswith(".zarr"):
                    zarr_path = os.path.join(root, dir_name)
                    relative_path = os.path.relpath(zarr_path, directory)
                    zarr_size = self.calculate_directory_size(zarr_path)
                    manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": zarr_size}

            dirs[:] = [d for d in dirs if not d.endswith(".zarr")]

        return manifest_entry

    def update_from_config(self, config: Dict, file_types: List[str] = [".tif", ".csv", ".h5ad"]) -> None:
        """
        Update the manifest based on the state of directories defined in the config.

        Args:
            config (Dict): Configuration dictionary with paths to directories.
            file_types (List[str]): List of file extensions to track.
        """
        for crunch_name, crunch_config in config["crunches"].items():
            project_dir = crunch_config["paths"]["project_dir"]

            if os.path.exists(project_dir):
                print(f"Updating manifest for {crunch_name}...")
                self.manifest[crunch_name] = self.generate_manifest_entry(project_dir, file_types)
            else:
                print(f"Directory not found for {crunch_name}: {project_dir}")

        self.save_manifest()
