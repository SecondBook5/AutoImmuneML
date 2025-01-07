# File: src/config/config_loader.py
import yaml  # Library for parsing YAML files
from typing import Any, Dict  # Type hints for function signatures


class ConfigLoader:
    """
    A utility class to load and access configuration settings from a YAML file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ConfigLoader with the path to the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Store the path to the YAML configuration file
        self.config_path = config_path
        # Load the configuration file into memory during initialization
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the YAML configuration file.

        Returns:
            Dict: The loaded configuration data.
        """
        try:
            # Open the YAML configuration file for reading
            with open(self.config_path, "r") as file:
                # Parse and return the contents as a dictionary
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Raise an error if the file is not found
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            # Raise an error if the YAML file is invalid or cannot be parsed
            raise ValueError(f"Error parsing YAML file: {e}")

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a global setting from the configuration.

        Args:
            key (str): The key of the global setting.
            default (Any): The default value to return if the key is not found (default: None).

        Returns:
            Any: The value of the global setting or the default value if not found.
        """
        # Access the "global" section of the configuration
        global_settings = self.config.get("global", {})
        # Check if the key exists in the global settings
        if key not in global_settings:
            # Print a warning and return the default value if the key is not found
            print(f"Warning: Global setting '{key}' not found. Returning default: {default}")
            return default
        # Return the value of the requested global setting
        return global_settings[key]

    def get_crunch_setting(self, crunch_name: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a setting for a specific Crunch.

        Args:
            crunch_name (str): The name of the Crunch (e.g., "crunch1").
            key (str): The key of the Crunch-specific setting.
            default (Any): The default value to return if the key is not found (default: None).

        Returns:
            Any: The value of the setting for the specified Crunch or the default value if not found.
        """
        # Access the settings for the specified Crunch
        crunch_settings = self.config.get("crunches", {}).get(crunch_name, {})
        # Check if the key exists in the Crunch-specific settings
        if key not in crunch_settings:
            # Print a warning and return the default value if the key is not found
            print(f"Warning: Setting '{key}' not found for {crunch_name}. Returning default: {default}")
            return default
        # Return the value of the requested setting
        return crunch_settings[key]

    def get_crunch_path(self, crunch_name: str, path_key: str, default: str = None) -> str:
        """
        Retrieve a path setting for a specific Crunch.

        Args:
            crunch_name (str): The name of the Crunch (e.g., "crunch1").
            path_key (str): The key of the path (e.g., "project_dir").
            default (str): The default path to return if the key is not found (default: None).

        Returns:
            str: The path value for the specified Crunch or the default path if not found.
        """
        # Access the "paths" section for the specified Crunch
        crunch_paths = self.get_crunch_setting(crunch_name, "paths", {})
        # Check if the path key exists in the paths section
        if path_key not in crunch_paths:
            # Print a warning and return the default path if the key is not found
            print(f"Warning: Path key '{path_key}' not found for {crunch_name}. Returning default: {default}")
            return default
        # Return the value of the requested path
        return crunch_paths[path_key]

    def get_training_setting(self, crunch_name: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a training-related setting for a specific Crunch.

        Args:
            crunch_name (str): The name of the Crunch (e.g., "crunch1").
            key (str): The key of the training setting (e.g., "batch_size").
            default (Any): The default value to return if the key is not found (default: None).

        Returns:
            Any: The value of the training setting or the default value if not found.
        """
        # Access the "training" section for the specified Crunch
        training_settings = self.get_crunch_setting(crunch_name, "training", {})
        # Check if the training key exists in the training section
        if key not in training_settings:
            # Print a warning and return the default value if the key is not found
            print(f"Warning: Training setting '{key}' not found for {crunch_name}. Returning default: {default}")
            return default
        # Return the value of the requested training setting
        return training_settings[key]

    def get_preprocessing_setting(self, crunch_name: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a preprocessing-related setting for a specific Crunch.

        Args:
            crunch_name (str): The name of the Crunch (e.g., "crunch1").
            key (str): The key of the preprocessing setting (e.g., "sample_size").
            default (Any): The default value to return if the key is not found (default: None).

        Returns:
            Any: The value of the preprocessing setting or the default value if not found.
        """
        # Access the "preprocessing" section for the specified Crunch
        preprocessing_settings = self.get_crunch_setting(crunch_name, "preprocessing", {})
        # Check if the preprocessing key exists in the preprocessing section
        if key not in preprocessing_settings:
            # Print a warning and return the default value if the key is not found
            print(f"Warning: Preprocessing setting '{key}' not found for {crunch_name}. Returning default: {default}")
            return default
        # Return the value of the requested preprocessing setting
        return preprocessing_settings[key]
