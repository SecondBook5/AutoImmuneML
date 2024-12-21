# Autoimmune Disease Machine Learning Challenge Data Downloader
# This script downloads and sets up the workspace for CrunchDAO challenges using the Crunch CLI.

# Import necessary libraries for file handling, YAML parsing, subprocess execution, and typing
import os
import yaml
import subprocess
from typing import Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load the project configuration from a YAML file
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
        # Define the path to the configuration file relative to the current script
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        # Check if the configuration file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Open and parse the configuration file
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")

# Function to validate the configuration structure
def validate_config(config: Dict):
    """
    Validate that the loaded configuration contains the required sections and keys.

    Args:
        config (dict): The loaded configuration.

    Raises:
        ValueError: If required sections or keys are missing.
    """
    required_sections = ["global", "crunches"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config.yaml: {section}")

    if "token_file" not in config["global"]:
        raise ValueError("Missing 'token_file' in global configuration.")

# Function to validate Crunch CLI installation
def validate_crunch_cli():
    """
    Validate that the Crunch CLI is installed and accessible.

    Raises:
        RuntimeError: If the Crunch CLI is not found or not executable.
    """
    try:
        subprocess.run(["crunch", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        raise RuntimeError("Crunch CLI is not installed or not in the system PATH.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Crunch CLI is installed but not functioning correctly.")

# Function to retrieve the authentication token from a file
def get_token(token_file: str, line_number: Optional[int] = None) -> str:
    """
    Retrieve the authentication token from a specified file.

    Args:
        token_file (str): The path to the token file.
        line_number (Optional[int]): The line number to read from the token file (1-based index).

    Returns:
        str: The token string used for authentication.

    Raises:
        FileNotFoundError: If the token file does not exist.
        ValueError: If the token file is empty or the specified line is invalid.
    """
    try:
        # Check if the token file exists
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Token file not found: {token_file}")

        # Read the content of the token file
        with open(token_file, "r") as file:
            lines = file.readlines()

            if line_number:
                if line_number > len(lines):
                    raise ValueError(f"Token file does not have line {line_number}.")
                token = lines[line_number - 1].strip()
            else:
                token = lines[0].strip()

            if not token:
                raise ValueError("Token file is empty or specified line is blank.")
            return token

    except Exception as e:
        raise RuntimeError(f"Error reading token file: {e}")

# Function to validate the output directory
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

        # Test write permissions by creating and deleting a temporary file
        test_file = os.path.join(output_dir, "test_file.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

    except PermissionError:
        raise RuntimeError(f"Output directory is not writable: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Error validating output directory: {e}")

# Function to download data using the Crunch CLI with retry logic
def download_data(competition_name: str, project_name: str, dataset_size: str, token: str, output_dir: str, retries: int = 3):
    """
    Setup a workspace directory and download the dataset using the Crunch CLI.

    Args:
        competition_name (str): The competition name (e.g., 'broad-1').
        project_name (str): The CrunchDAO project name (e.g., 'autoimmune-crunch1').
        dataset_size (str): Dataset size to download ('default' or 'large').
        token (str): Authentication token for CrunchDAO.
        output_dir (str): Directory where the workspace will be set up.
        retries (int): Number of retry attempts for the command.

    Raises:
        RuntimeError: If the Crunch CLI command fails after retries.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Validate the output directory before starting the download
            validate_output_directory(output_dir)

            # Construct the Crunch CLI command
            command = [
                "crunch", "setup",
                competition_name, project_name,
                output_dir,
                "--token", token,
                "--size", dataset_size
            ]
            print("Running command:", " ".join(command))

            # Execute the command using subprocess
            subprocess.run(command, check=True)
            print("Workspace setup completed successfully.")
            return  # Exit the function on success

        except subprocess.CalledProcessError as e:
            print(f"Error during workspace setup (attempt {attempt + 1}): {e}")
            attempt += 1
            time.sleep(2)  # Wait before retrying

    raise RuntimeError(f"Crunch CLI failed after {retries} attempts.")

# Function to handle parallel downloads within a single Crunch
def process_crunch(name: str, config: Dict, token_file: str):
    """
    Process a single Crunch task with parallel retries.

    Args:
        name (str): The name of the Crunch (e.g., 'crunch1').
        config (dict): The configuration for the Crunch.
        token_file (str): The path to the token file.
    """
    try:
        print(f"Processing {name}...")
        token = get_token(token_file, line_number=int(name[-1]))

        # Simulate parallel retries (e.g., downloading sub-tasks within the same Crunch)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_data, config["crunch_type"], config["name"], config["dataset_size"], token, config["paths"]["raw_dir"], retries=1)
                for _ in range(3)  # Simulating multiple parallel download attempts or sub-tasks
            ]
            for future in as_completed(futures):
                future.result()

    except Exception as e:
        print(f"Error processing {name}: {e}")

# Main script execution
def main():
    """
    Main script execution:
    - Load the configuration from config.yaml.
    - Validate the configuration structure.
    - Check Crunch CLI availability.
    - Retrieve the authentication token.
    - Validate the output directory.
    - Use Crunch CLI to download the specified dataset or all datasets in parallel.
    """
    try:
        # Step 1: Load and validate the configuration
        config = load_config()
        validate_config(config)

        # Step 2: Validate Crunch CLI installation
        validate_crunch_cli()

        # Step 3: Select the desired Crunch configuration or all
        crunch_name = input("Enter the Crunch name (e.g., 'crunch1', 'crunch2', 'crunch3', or 'all'): ").strip()

        global_config = config["global"]
        crunches = config["crunches"]

        if crunch_name == "all":
            with ThreadPoolExecutor() as outer_executor:
                outer_futures = [
                    outer_executor.submit(process_crunch, name, crunch_config, global_config["token_file"])
                    for name, crunch_config in crunches.items()
                ]
                for future in as_completed(outer_futures):
                    future.result()
        elif crunch_name in crunches:
            process_crunch(crunch_name, crunches[crunch_name], global_config["token_file"])
        else:
            raise ValueError(f"Invalid Crunch name: {crunch_name}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
