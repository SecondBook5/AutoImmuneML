# File: src/utils/manifest_manager.py

import os  # Import os module for handling file paths
import json  # Import json module for reading and writing JSON data
from typing import Dict  # Import Dict type hint for function annotations


class ManifestManager:
    """
    Manages loading and saving of the manifest file, ensuring structured and consistent access.
    """

    def __init__(self, manifest_path: str):
        """
        Initialize the manager with the manifest file path.

        Args:
            manifest_path (str): Path to the manifest file.
        """
        # Save the path to the manifest file for future operations
        self.manifest_path = manifest_path

    def load_manifest(self) -> Dict:
        """
        Load the manifest from the file.

        Returns:
            Dict: The loaded manifest as a dictionary. If the file doesn't exist, returns an empty dictionary.
        """
        # Check if the manifest file exists
        if not os.path.exists(self.manifest_path):
            # Print a message if the manifest file does not exist and return an empty dictionary
            print(f"[INFO] Manifest file not found at {self.manifest_path}. Starting with an empty manifest.")
            return {}

        try:
            # Open the manifest file in read mode
            with open(self.manifest_path, "r") as file:
                # Parse the JSON content of the file
                manifest = json.load(file)
                # Ensure the loaded manifest is a dictionary
                if not isinstance(manifest, dict):
                    raise ValueError("Manifest content is not a valid dictionary.")
                # Return the loaded manifest as a dictionary
                return manifest
        except json.JSONDecodeError:
            # Handle and raise an error if the JSON is malformed
            raise ValueError(f"[ERROR] Malformed JSON in manifest file: {self.manifest_path}.")
        except Exception as e:
            # Catch and raise any other unexpected errors during file reading
            raise RuntimeError(f"[ERROR] Unexpected error loading manifest: {e}")

    def save_manifest(self, manifest: Dict):
        """
        Save the manifest to the file.

        Args:
            manifest (Dict): The manifest data to save.
        """
        # Ensure the manifest is a dictionary before proceeding
        if not isinstance(manifest, dict):
            raise ValueError("[ERROR] Provided manifest is not a valid dictionary.")

        try:
            # Open the manifest file in write mode
            with open(self.manifest_path, "w") as file:
                # Serialize the dictionary and save it as JSON with indentation for readability
                json.dump(manifest, file, indent=4)
            # Print a success message after the manifest is saved
            print(f"[INFO] Manifest successfully saved to {self.manifest_path}.")
        except PermissionError:
            # Handle and raise an error if there are insufficient permissions to write to the file
            raise PermissionError(f"[ERROR] Permission denied when writing to manifest file: {self.manifest_path}.")
        except Exception as e:
            # Catch and raise any other unexpected errors during file writing
            raise RuntimeError(f"[ERROR] Unexpected error saving manifest: {e}")
