# File: src/utils/path_validator.py
# This script validates and ensures the directory structure and required files defined in the configuration file.

import os  # For file and directory operations
import yaml  # For parsing YAML configuration files
import logging  # For structured logging
from typing import Dict  # For more explicit type annotations

class PathValidator:
    """
    A utility class to validate and ensure the directory structure and required files
    defined in the configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the PathValidator with the path to the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Store the path to the configuration file
        self.config_path = config_path
        # Load the configuration file and parse its contents
        self.config = self._load_config()
        # Set up a logger for structured output
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Setup a logger for structured output.

        Returns:
            logging.Logger: Configured logger instance.
        """
        # Create a logger instance
        logger = logging.getLogger("PathValidator")
        # Define a stream handler for logging to console
        handler = logging.StreamHandler()
        # Set the log format to include level and message
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(handler)
        # Set the default logging level to INFO
        logger.setLevel(logging.INFO)
        return logger

    def _load_config(self) -> Dict:
        """
        Load the configuration from a YAML file.

        Returns:
            Dict: Parsed YAML configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is invalid or cannot be parsed.
        """
        # Check if the configuration file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            # Open the YAML file and parse its contents
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            # Raise a value error if the YAML file cannot be parsed
            raise ValueError(f"Error parsing YAML file: {e}")

    def ensure_path(self, path: str, is_file: bool = False) -> bool:
        """
        Ensure a path is valid: check for files or create directories as needed.

        Args:
            path (str): Path to check or create.
            is_file (bool): Whether the path is a file (default: False).

        Returns:
            bool: True if the path is valid or created, False otherwise.
        """
        # Check if the path exists
        if os.path.exists(path):
            # If the path is supposed to be a file but is not, log an error
            if is_file and not os.path.isfile(path):
                self.logger.error(f"Path exists but is not a file: {path}")
                return False
            # If the path is supposed to be a directory but is not, log an error
            if not is_file and not os.path.isdir(path):
                self.logger.error(f"Path exists but is not a directory: {path}")
                return False
            # Log a success message if the path is valid
            self.logger.info(f"Valid {'file' if is_file else 'directory'}: {path}")
            return True

        # If the path is supposed to be a file but does not exist, log a warning
        if is_file:
            self.logger.warning(f"Missing required file: {path}")
            return False

        # If the path is supposed to be a directory, create it
        os.makedirs(path, exist_ok=True)
        # Log a success message after creating the directory
        self.logger.info(f"Created directory: {path}")
        return True

    def validate_paths(self) -> None:
        """
        Validate and ensure all paths in the configuration file exist.
        """
        # Log the start of global path validation
        self.logger.info("Validating global paths...")
        # Iterate through global paths in the configuration and validate each
        for key, path in self.config.get("global", {}).items():
            self.ensure_path(path, is_file=key.endswith("_file"))

        # Iterate through Crunch-specific configurations
        for crunch_name, crunch_config in self.config.get("crunches", {}).items():
            # Log the start of validation for the specific Crunch
            self.logger.info(f"Validating paths for {crunch_name}...")
            # Validate each path in the Crunch-specific configuration
            for key, path in crunch_config.get("paths", {}).items():
                self.ensure_path(path, is_file=key.endswith("_file"))

if __name__ == "__main__":
    import argparse  # For parsing command-line arguments

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Validate project paths based on the configuration file.")
    # Add an argument for specifying the configuration file path
    parser.add_argument(
        "--config", type=str, default="../../config.yaml", help="Path to the configuration file."
    )
    args = parser.parse_args()

    try:
        # Create an instance of PathValidator with the specified config file
        validator = PathValidator(config_path=args.config)
        # Validate all paths in the configuration
        validator.validate_paths()
    except Exception as e:
        # Log any unexpected errors during execution
        print(f"[ERROR] {e}")
