# File: AutoImmuneML/crunch1_project/src/crunch_data_downloader.py
import os
import yaml
import subprocess
from typing import Dict


def load_config() -> Dict:
    """
    Load the project configuration from config.yaml.

    Returns:
        dict: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config.yaml file does not exist.
        ValueError: If the YAML file has invalid syntax.
    """
    try:
        # Build the path to the configuration file (config.yaml)
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )

        # Check if the config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Open and parse the config.yaml file into a dictionary
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")


def get_token(token_file: str) -> str:
    """
    Retrieve the authentication token from a specified file.

    Args:
        token_file (str): The path to the token file.

    Returns:
        str: The token string used for authentication.

    Raises:
        FileNotFoundError: If the token file does not exist.
        ValueError: If the token file is empty.
    """
    try:
        # Check if the token file exists
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token file not found: {token_file}")

        # Read the content of the token file
        with open(token_file, "r") as file:
            token = file.read().strip()

            # Validate that the token is not empty
            if not token:
                raise ValueError("Token file is empty.")
            return token

    except Exception as e:
        raise RuntimeError(f"Error reading token file: {e}")


def validate_output_directory(output_dir: str):
    """
    Validate that the output directory exists and is writable.

    Args:
        output_dir (str): Directory where the data will be saved.

    Raises:
        RuntimeError: If the directory is not writable or the filesystem is incompatible.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Test write permissions
        test_file = os.path.join(output_dir, "test_file.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

    except PermissionError:
        raise RuntimeError(f"Output directory is not writable: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Error validating output directory: {e}")


def download_data(project_name: str, dataset_size: str, token: str, output_dir: str):
    """
    Download the dataset using the Crunch CLI and ensure it ends up in the correct location.

    Args:
        project_name (str): The CrunchDAO project name.
        dataset_size (str): Dataset size to download ('default' or 'large').
        token (str): Authentication token for CrunchDAO.
        output_dir (str): Directory where the data will be saved.

    Raises:
        RuntimeError: If the Crunch CLI command fails.
    """
    try:
        # Validate the output directory
        validate_output_directory(output_dir)

        # Temporarily change working directory to /mnt/d (external hard drive root)
        os.chdir("/mnt/d/AutoImmuneML")

        # Construct the Crunch CLI command
        command = f"crunch setup --size {dataset_size} broad-1 {project_name} --token {token}"
        print("Running command:", command)

        # Execute the command using subprocess
        subprocess.run(command, shell=True, check=True)
        print("Data download completed successfully.")

        # Check for 'data' directory and move its contents to the raw_dir
        data_dir = os.path.join(os.getcwd(), "data")
        if os.path.exists(data_dir):
            print("Moving downloaded data to the specified raw directory...")
            for item in os.listdir(data_dir):
                src_path = os.path.join(data_dir, item)
                dst_path = os.path.join(output_dir, item)
                os.rename(src_path, dst_path)
            print("Data moved successfully.")

            # Clean up the empty 'data' directory
            os.rmdir(data_dir)

        # Clean up the 'resources' directory if it exists
        resources_dir = os.path.join(os.getcwd(), "resources")
        if os.path.exists(resources_dir):
            os.rmdir(resources_dir)
            print("Cleaned up unnecessary 'resources' directory.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Crunch CLI failed with error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during data download: {e}")



if __name__ == "__main__":
    """
    Main script execution:
    - Load the configuration from config.yaml.
    - Retrieve the authentication token.
    - Validate the output directory.
    - Use Crunch CLI to download the specified dataset.
    """
    try:
        # Step 1: Load project configuration from config.yaml
        config = load_config()

        # Step 2: Retrieve the token for authentication
        token_file = config["paths"]["token_file"]
        token = get_token(token_file)

        # Step 3: Extract project parameters from the configuration
        project_name = config["project"]["name"]
        dataset_size = config["project"]["dataset_size"]
        raw_dir = config["paths"]["raw_dir"]

        # Step 4: Download the data using Crunch CLI
        download_data(project_name, dataset_size, token, raw_dir)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
