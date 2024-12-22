# File: path_validator.py
import os
import yaml
from typing import Dict
from utils.path_utils import resolve_path, get_global_variables, merge_variables


def load_config() -> Dict:
    """
    Load the configuration from the config.yaml file.

    Returns:
        dict: Parsed YAML configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config.yaml file does not exist.
        ValueError: If the YAML file is invalid or cannot be parsed.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")


def ensure_directory_or_file(path: str):
    """
    Ensure that a path is valid. If it is a file, validate it exists.
    If it is a directory, create it if necessary.

    Args:
        path (str): Path to check or create.

    Raises:
        RuntimeError: If the path cannot be created or validated.
    """
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"[✔] File exists: {path}")
            elif os.path.isdir(path):
                print(f"[✔] Directory exists: {path}")
        else:
            print(f"[✘] Path does NOT exist: {path}. Attempting to create it.")
            os.makedirs(path, exist_ok=True)
            print(f"[✔] Directory created: {path}")
    except Exception as e:
        raise RuntimeError(f"Error ensuring path '{path}': {e}")


def validate_paths(config: Dict):
    """
    Validate all key paths specified in the config.yaml file.

    Args:
        config (dict): Parsed configuration dictionary.
    """
    print("\n--- Validating Project Directories ---\n")

    try:
        # Extract global variables
        global_vars = get_global_variables(config)
        crunches = config.get("crunches", {})

        # Validate global paths
        for description, path in config["global"].items():
            if description.endswith("_dir") or description == "token_file":
                resolved_path = resolve_path(path, global_vars)
                print(f"[DEBUG] Resolving global path for '{description}': {resolved_path}")
                ensure_directory_or_file(resolved_path)

        # Validate paths for each Crunch
        for crunch_name, crunch_config in crunches.items():
            print(f"\n--- Validating Paths for {crunch_name} ---")

            # Merge global and crunch-specific variables
            crunch_vars = {
                key: resolve_path(value, global_vars)
                for key, value in crunch_config["paths"].items()
            }
            all_vars = merge_variables(global_vars, crunch_vars)

            # Resolve each path dynamically and validate
            for description, path in crunch_config["paths"].items():
                resolved_path = resolve_path(path, all_vars)
                print(f"[DEBUG] Resolving path for '{description}' in {crunch_name}: {resolved_path}")
                ensure_directory_or_file(resolved_path)

    except KeyError as e:
        print(f"Missing key in configuration file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during path validation: {e}")

    print("\n--- Directory Validation Complete ---\n")


if __name__ == "__main__":
    """
    Main script execution: Load the configuration file and validate all paths.
    """
    try:
        config = load_config()
        validate_paths(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
