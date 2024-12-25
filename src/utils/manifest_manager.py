# File: src/utils/manifest_manager.py
import os  # Module for working with file paths
import json  # Module for reading and writing JSON data
from typing import Dict  # Type hints for dictionaries


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
        # Store the path to the manifest file for future operations
        self.manifest_path = manifest_path

    def load_manifest(self) -> Dict:
        """
        Load the manifest from the file.

        Returns:
            Dict: The loaded manifest as a dictionary. If the file doesn't exist, returns an empty dictionary.
        """
        # Check if the manifest file exists
        if not os.path.exists(self.manifest_path):
            # Inform the user that a new manifest will be created
            print(f"Manifest file not found at {self.manifest_path}. Starting with an empty manifest.")
            return {}  # Return an empty dictionary as the default manifest

        try:
            # Open and read the manifest JSON file
            with open(self.manifest_path, "r") as file:
                return json.load(file)  # Parse and return the contents as a dictionary
        except json.JSONDecodeError:
            # Handle invalid JSON formatting gracefully
            raise ValueError(f"Error parsing the manifest file at {self.manifest_path}. Ensure it is valid JSON.")
        except Exception as e:
            # Handle unexpected errors during file reading
            raise RuntimeError(f"Unexpected error while loading the manifest file: {e}")

    def save_manifest(self, manifest: Dict):
        """
        Save the manifest to the file.

        Args:
            manifest (Dict): The manifest data to save.
        """
        try:
            # Write the provided manifest dictionary to the JSON file
            with open(self.manifest_path, "w") as file:
                json.dump(manifest, file, indent=4)  # Use indentation for readable formatting
            # Notify the user that the manifest was saved successfully
            print(f"Manifest saved to {self.manifest_path}.")
        except PermissionError:
            # Handle file permission issues
            raise PermissionError(f"Permission denied while saving the manifest to {self.manifest_path}.")
        except Exception as e:
            # Handle other unexpected errors during file writing
            raise RuntimeError(f"Unexpected error while saving the manifest file: {e}")
