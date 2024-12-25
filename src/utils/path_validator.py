# File: src/utils/path_validator.py
# This script validates and ensures the directory structure and required files defined in the configuration file.

import os  # For file and directory operations
import yaml  # For parsing YAML configuration files
from typing import Dict  # For more explicit type annotations


def load_config(config_path: str) -> Dict:
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: Parsed YAML configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is invalid or cannot be parsed.
    """
    try:
        # Check if the configuration file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        # Open and parse the YAML file into a dictionary
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except FileNotFoundError as e:
        # Re-raise the file not found error for upstream handling
        raise e

    except yaml.YAMLError as e:
        # Raise an error for YAML parsing issues
        raise ValueError(f"Error parsing YAML file: {e}")

    except Exception as e:
        # Raise a generic error for unexpected issues
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")


def ensure_path(path: str, is_file: bool = False) -> None:
    """
    Ensure a path is valid: check for files or create directories as needed.

    Args:
        path (str): Path to check or create.
        is_file (bool): Whether the path is a file (default: False).

    How It Works:
        - For files: Logs a warning if missing.
        - For directories: Creates them if they do not exist.
    """
    try:
        # Check if the path already exists
        if os.path.exists(path):
            # Check if the existing path matches its expected type
            if is_file and not os.path.isfile(path):
                print(f"[✘] Path exists but is not a file: {path}")
            elif not is_file and not os.path.isdir(path):
                print(f"[✘] Path exists but is not a directory: {path}")
            else:
                # Log a success message if the path is valid
                print(f"[✔] {'File' if is_file else 'Directory'} exists: {path}")
        else:
            # Handle missing files
            if is_file:
                print(f"[⚠] Required file missing: {path}. Skipping...")
            else:
                # Attempt to create missing directories
                print(f"[✘] Directory does NOT exist: {path}. Attempting to create it.")
                os.makedirs(path, exist_ok=True)
                print(f"[✔] Directory created: {path}")
    except Exception as e:
        # Log any errors encountered during path validation
        print(f"[⚠] Error ensuring path '{path}': {e}. Continuing...")


def validate_paths(config: Dict):
    """
    Validate and ensure all paths in the configuration file exist.

    Args:
        config (Dict): Configuration dictionary with absolute paths.

    How It Works:
        - Iterates through global and Crunch-specific paths.
        - Ensures files exist and directories are created if missing.
    """
    print("\n--- Validating Project Directories and Files ---\n")

    try:
        # Validate paths in the global configuration section
        print("[INFO] Validating global paths...")
        for key, path in config["global"].items():
            # Check if the key corresponds to a file or directory
            is_file = key.endswith("_file")
            ensure_path(path, is_file=is_file)

        # Validate Crunch-specific paths
        for crunch_name, crunch_config in config["crunches"].items():
            print(f"\n--- Validating Paths for {crunch_name} ---")
            for key, path in crunch_config["paths"].items():
                # Determine if the path corresponds to a file or directory
                is_file = key.endswith("_file")
                ensure_path(path, is_file=is_file)

    except KeyError as e:
        # Handle missing keys in the configuration
        print(f"[⚠] Missing key in configuration file: {e}. Continuing...")

    except Exception as e:
        # Log any unexpected errors during validation
        print(f"[⚠] An unexpected error occurred during path validation: {e}. Continuing...")

    print("\n--- Directory and File Validation Complete ---\n")


if __name__ == "__main__":
    """
    Main script execution: Load the configuration file and validate all paths.
    """
    try:
        # Define the path to the configuration file
        config_file_path = "../../config.yaml"

        # Load the configuration file into a dictionary
        config = load_config(config_file_path)

        # Validate all paths based on the loaded configuration
        validate_paths(config)

    except FileNotFoundError as e:
        # Handle missing configuration file errors
        print(f"Error: {e}")

    except ValueError as e:
        # Handle YAML parsing errors
        print(f"Configuration Error: {e}")

    except Exception as e:
        # Handle any unexpected errors
        print(f"An unexpected error occurred: {e}")
