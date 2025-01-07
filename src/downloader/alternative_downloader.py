# Autoimmune Disease Machine Learning Challenge Data Downloader
# File: src/downloader/alternative_downloader.py
# This script downloads and sets up the workspace for CrunchDAO challenges using the Crunch CLI.

import os
import json
import subprocess
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the path to the manifest file
MANIFEST_FILE = "../manifest.json"

# Function to load the project configuration from a YAML file
def load_config() -> Dict:
    """
    Load the project configuration from config.yaml.

    Returns:
        Dict: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config.yaml file does not exist.
        ValueError: If the YAML file has invalid syntax.
        RuntimeError: For unexpected errors during loading.
    """
    import yaml
    try:
        # Define the path to the configuration file
        config_path = "../../config.yaml"

        # Check if the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Open and parse the YAML file
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    except yaml.YAMLError as e:
        # Raise a ValueError for YAML parsing errors
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        # Raise a generic runtime error for unexpected issues
        raise RuntimeError(f"Unexpected error while loading config.yaml: {e}")

# Function to validate the configuration structure
def validate_config(config: Dict) -> None:
    """
    Validate the structure of the configuration file.

    Args:
        config (Dict): The configuration dictionary.

    Raises:
        ValueError: If the configuration is missing required keys or sections.
    """
    # Define the required sections
    required_sections = ["global", "crunches"]

    # Check if each required section exists in the config
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Check if the 'token_file' key exists in the global configuration
    if "token_file" not in config["global"]:
        raise ValueError("Missing 'token_file' in global configuration.")

# Function to validate Crunch CLI installation
def validate_crunch_cli()-> None:
    """
    Ensure that the Crunch CLI is installed and accessible.

    Raises:
        RuntimeError: If the CLI is not installed or is not functioning.
    """
    try:
        # Run a test command to check if the Crunch CLI is installed
        subprocess.run(["crunch", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        # Raise an error if the Crunch CLI is not found
        raise RuntimeError("Crunch CLI is not installed or not in the system PATH.")
    except subprocess.CalledProcessError:
        # Raise an error if the Crunch CLI command fails unexpectedly
        raise RuntimeError("Crunch CLI is installed but not functioning correctly.")

# Function to retrieve the authentication token from a file
def get_token(token_file: str, line_number: int) -> str:
    """
    Retrieve the CrunchDAO authentication token from a file.

    Args:
        token_file (str): Path to the token file.
        line_number (int): Line number for token (1-based).

    Returns:
        str: The authentication token.
    """
    # Check if the token file exists
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Token file not found: {token_file}")

    # Open the token file and read its lines
    with open(token_file, "r") as file:
        lines = file.readlines()

        # Validate the requested line number
        if line_number > len(lines):
            raise ValueError(f"Token file does not have line {line_number}.")

        # Return the token at the specified line number
        return lines[line_number - 1].strip()

# Function to load the manifest
def load_manifest() -> Dict:
    """
    Load the manifest file.

    Returns:
        Dict: Loaded manifest or an empty dictionary if not found.
    """
    # Check if the manifest file exists
    if os.path.exists(MANIFEST_FILE):
        # Load and return the manifest as a dictionary
        with open(MANIFEST_FILE, "r") as file:
            return json.load(file)
    # Return an empty dictionary if the manifest does not exist
    return {}

# Function to save the manifest
def save_manifest(manifest: Dict) -> None:
    """
    Save the manifest file.

    Args:
        manifest (Dict): Manifest data to save.
    """
    # Write the manifest dictionary to the manifest file as JSON
    with open(MANIFEST_FILE, "w") as file:
        json.dump(manifest, file, indent=4)

# Function to update the manifest
def update_manifest(manifest: Dict, crunch_name: str, file_path: str, status: str, size: int = 0) -> None:
    """
    Update the manifest with the status of a file.

    Args:
        manifest (Dict): The current manifest data.
        crunch_name (str): Name of the Crunch being updated.
        file_path (str): Path of the file being updated.
        status (str): Status of the file (e.g., "downloaded").
        size (int): Size of the file in bytes.
    """
    # Initialize the crunch entry in the manifest if not already present
    if crunch_name not in manifest:
        manifest[crunch_name] = {}

    # Update the file's entry in the manifest
    manifest[crunch_name][file_path] = {
        "status": status,
        "size_bytes": size
    }


# Function to check if a file or directory is downloaded
def is_downloaded(manifest: Dict, crunch_name: str, file_path: str) -> bool:
    """
    Check if a file or directory is already downloaded.

    Args:
        manifest (Dict): The current manifest data.
        crunch_name (str): Name of the Crunch.
        file_path (str): Path of the file to check.

    Returns:
        bool: True if the file is downloaded, False otherwise.
    """
    return manifest.get(crunch_name, {}).get(file_path, {}).get("status") == "downloaded"

# Function to download data using the Crunch CLI
def download_data(competition_name: str, project_name: str, dataset_size: str, token: str, output_dir: str, dry_run: bool) -> bool:
    """
    Download data for a crunch using the Crunch CLI.

    Args:
        competition_name (str): Competition name.
        project_name (str): Project name.
        dataset_size (str): Dataset size.
        token (str): Authentication token.
        output_dir (str): Output directory.
        dry_run (bool): Whether to simulate the download.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    # Construct the Crunch CLI command
    command = [
        "crunch", "setup",
        competition_name, project_name,
        output_dir,
        "--token", token,
        "--size", dataset_size
    ]

    # Print the command for dry-run mode and return success
    if dry_run:
        print(f"[DRY-RUN] Would run command: {' '.join(command)}")
        return True

    try:
        # Execute the Crunch CLI command
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        # Log the error on download failure
        print(f"[✘] Download failed: {e}")
        return False

# Function to process a single crunch
def process_crunch(crunch_name: str, config: Dict, token: str, manifest: Dict, dry_run: bool) -> None:
    """
    Process and download data for a specific crunch.

    Args:
        crunch_name (str): Crunch name.
        config (Dict): Crunch-specific configuration.
        token (str): Authentication token.
        manifest (Dict): Manifest data.
        dry_run (bool): Whether to simulate the download.
    """
    # Get the project directory from the configuration
    project_dir = config["paths"]["project_dir"]

    # Check if the project directory is already downloaded
    if is_downloaded(manifest, crunch_name, project_dir):
        print(f"[✔] {crunch_name} is already downloaded. Skipping.")
        return

    print(f"Processing {crunch_name}...")

    # Define the download task
    def download_task() -> bool:
        """
        Download the data for the current crunch.

        Returns:
            bool: True if the download was successful, False otherwise.
        """
        return download_data(
            config["crunch_type"],
            config["name"],
            config["dataset_size"],
            token,
            project_dir,
            dry_run
        )

    # Execute the download task in parallel retries
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_task) for _ in range(3)]
        success = all(f.result() for f in as_completed(futures))

    # Update the manifest with the result of the download
    if success:
        size = os.path.getsize(project_dir) if os.path.exists(project_dir) else 0
        update_manifest(manifest, crunch_name, project_dir, "downloaded", size)
        print(f"[✔] {crunch_name} downloaded successfully and manifest updated.")
    else:
        print(f"[✘] Failed to download {crunch_name}.")


# Main script execution
def main():
    """
    Main script for managing downloads for CrunchDAO challenges.

    This script reads the configuration, validates it, and processes the Crunches
    """
    # Record the start time for runtime measurement
    start_time = time.time()

    # Prompt the user to enable dry-run mode
    dry_run = input("Enable dry-run mode? (y/n): ").strip().lower() == "y"
    try:
        # Load and validate the configuration
        config = load_config()
        validate_config(config)

        # Ensure the Crunch CLI is installed
        validate_crunch_cli()

        # Load the manifest to track downloads
        manifest = load_manifest()
        token_file = config["global"]["token_file"]

        # Prompt the user to select a crunch or all
        crunch_selection = input("Enter the Crunch name (e.g., 'crunch1', 'crunch2', 'crunch3', or 'all'): ").strip()
        if crunch_selection == "all":
            # Process all crunches in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_crunch, name, crunch_config, get_token(token_file, idx + 1), manifest, dry_run)
                    for idx, (name, crunch_config) in enumerate(config["crunches"].items())
                ]
                for future in as_completed(futures):
                    future.result()
        elif crunch_selection in config["crunches"]:
            # Process a specific crunch
            process_crunch(
                crunch_selection,
                config["crunches"][crunch_selection],
                get_token(token_file, list(config["crunches"].keys()).index(crunch_selection) + 1),
                manifest,
                dry_run
            )
        else:
            print(f"Invalid selection: {crunch_selection}")

        # Save the manifest after processing
        save_manifest(manifest)

    except Exception as e:
        # Print an error message if any exception occurs
        print(f"[ERROR] {e}")
    finally:
        # Print the total runtime of the script
        runtime = time.time() - start_time
        print(f"Total runtime: {runtime:.2f} seconds")


# Run the script
if __name__ == "__main__":
    main()
