# File: src/utils/path_validator.py
# This script validates and ensures the directory structure and required files defined in the configuration file.

import os  # For file and directory operations
import yaml  # For parsing YAML configuration files
import logging  # For structured logging
from typing import Dict  # For more explicit type annotations
from src.validators.zarr_validator import ZARRValidator  # For validating Zarr datasets

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
        Setup a logger for structured output. Reuses the logger if already set up.

        Returns:
            logging.Logger: Configured logger instance.
        """
        # Create or get the logger for "PathValidator"
        logger = logging.getLogger("PathValidator")
        # Ensure no duplicate handlers are added
        if not logger.hasHandlers():
            # Define a stream handler for console output
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

    def ensure_path(self, path: str, is_file: bool = False, is_zarr: bool = False) -> bool:
        """
        Ensure a path is valid: check for files, directories, or `.zarr` datasets.

        Args:
            path (str): Path to check or create.
            is_file (bool): Whether the path is expected to be a file (default: False).
            is_zarr (bool): Whether the path is expected to be a `.zarr` directory (default: False).

        Returns:
            bool: True if the path is valid or created, False otherwise.
        """
        # Check if the path exists
        if os.path.exists(path):
            # Validate if the path is a file but is not
            if is_file and not os.path.isfile(path):
                self.logger.error(f"Path exists but is not a file: {path}")
                return False
            # Validate if the path is a `.zarr` directory but is not
            if is_zarr and (not os.path.isdir(path) or not path.endswith(".zarr")):
                self.logger.error(f"Path exists but is not a valid `.zarr` directory: {path}")
                return False
            # Validate if the path is a directory but is not
            if not is_file and not is_zarr and not os.path.isdir(path):
                self.logger.error(f"Path exists but is not a directory: {path}")
                return False
            # Log success if the path is valid
            self.logger.info(f"Valid {'file' if is_file else 'zarr dataset' if is_zarr else 'directory'}: {path}")
            return True

        # Warn if the required file is missing
        if is_file:
            self.logger.warning(f"Missing required file: {path}")
            return False

        # Warn if the required `.zarr` directory is missing
        if is_zarr:
            self.logger.error(f"Missing required `.zarr` directory: {path}")
            return False

        # Create the directory if it's missing
        os.makedirs(path, exist_ok=True)
        # Log a success message after creating the directory
        self.logger.info(f"Created directory: {path}")
        return True

    def validate_paths(self) -> None:
        """
        Validate and ensure all paths in the configuration file exist.
        Skips non-path keys and validates only paths.
        """
        # Log the start of global path validation
        self.logger.info("Validating global paths...")

        # Validate global paths
        for key, path in self.config.get("global", {}).items():
            # Skip non-path keys (e.g., max_workers, batch_size)
            if not isinstance(path, (str, os.PathLike)):
                self.logger.debug(f"Ignoring non-path key: {key}")
                continue
            # Validate the path (file or directory)
            self.ensure_path(path, is_file=key.endswith("_file"))

        # Validate paths for each Crunch
        for crunch_name, crunch_config in self.config.get("crunches", {}).items():
            # Log the start of validation for this Crunch
            self.logger.info(f"Validating paths for {crunch_name}...")
            # Validate each path in the Crunch-specific configuration
            for key, path in crunch_config.get("paths", {}).items():
                # Skip non-path keys (e.g., batch_size)
                if not isinstance(path, (str, os.PathLike)):
                    self.logger.debug(f"Ignoring non-path key: {key}")
                    continue

                # Determine if this path is a `.zarr` directory or file
                is_zarr = key.endswith(".zarr") or "_zarr" in key
                # Validate the path
                if self.ensure_path(path, is_file=key.endswith("_file"), is_zarr=is_zarr) and is_zarr:
                    self.logger.info(f"Validating `.zarr` structure for {path}...")
                    # Use ZARRValidator to validate `.zarr` structure
                    validator = ZARRValidator(path)
                    validation_result = validator.validate()
                    if validation_result["status"] == "invalid":
                        self.logger.error(f"Validation failed for `.zarr`: {path}")
                        for error in validation_result["errors"]:
                            self.logger.error(f"  - {error}")


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
