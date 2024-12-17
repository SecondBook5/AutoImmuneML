import os
import yaml
from typing import Dict


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
        # Build the path to config.yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml")
        # Verify that the config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found at: {config_path}")

        # Open and parse the YAML file
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")


def check_directory(path: str, description: str):
    """
    Check if a directory or file exists and print its status.

    Args:
        path (str): Path to verify.
        description (str): A human-readable description of the directory/file.
    
    Returns:
        bool: True if the path exists, False otherwise.
    """
    try:
        # Defensive check for invalid path inputs
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Invalid path provided for '{description}'.")

        # Verify if the path exists
        if os.path.exists(path):
            print(f"[✔] {description}: Path exists -> {path}")
            return True
        else:
            print(f"[✘] {description}: Path does NOT exist -> {path}")
            return False
    except Exception as e:
        print(f"Error checking path '{description}': {e}")
        return False


def validate_paths(config: Dict):
    """
    Validate all key paths specified in the config.yaml file.

    Args:
        config (dict): Parsed configuration dictionary.
    """
    print("\n--- Validating Project Directories ---\n")

    try:
        # Retrieve paths from the configuration file
        paths_to_check = {
            "Raw Data Directory": config["paths"].get("raw_dir"),
            "Interim Data Directory": config["paths"].get("interim_dir"),
            "Train Data Directory": config["paths"].get("train_dir"),
            "Test Data Directory": config["paths"].get("test_dir"),
            "Predictions Directory": config["paths"].get("predictions_dir"),
            "CrunchDAO Token File": config["paths"].get("token_file"),
            "Models Directory": config["paths"].get("models_dir"),
            "Results Directory": config["paths"].get("results_dir"),
            "Source Code Directory": config["paths"].get("src_dir"),
            "Logs Directory": config["paths"].get("logs_dir")
        }

        # Loop through each path and validate
        for description, path in paths_to_check.items():
            # Defensive check for missing keys in the config file
            if path is None:
                print(f"[✘] {description}: Path is missing in config.yaml")
                continue
            check_directory(path, description)

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
        # Load the configuration file
        config = load_config()
        # Validate paths based on the configuration
        validate_paths(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
