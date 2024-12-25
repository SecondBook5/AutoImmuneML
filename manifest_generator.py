# File: manifest_generator.py
import os
import json
from typing import Dict, List

# Define the default manifest file path
MANIFEST_FILE = "manifest.json"

# Function to load the manifest if it exists
def load_manifest() -> Dict:
    """
    Load the manifest file if it exists, or return an empty dictionary.

    Returns:
        Dict: The current manifest data, or an empty dictionary if no manifest exists.
    """
    # Check if the manifest file exists
    if os.path.exists(MANIFEST_FILE):
        # Open and read the manifest file
        with open(MANIFEST_FILE, "r") as file:
            return json.load(file)
    # Return an empty dictionary if the manifest file does not exist
    return {}

# Function to save the manifest data to a file
def save_manifest(manifest: Dict) -> None:
    """
    Save the manifest data to a file.

    Args:
        manifest (Dict): The manifest data to save.
    """
    # Write the manifest data as JSON to the manifest file
    with open(MANIFEST_FILE, "w") as file:
        json.dump(manifest, file, indent=4)

# Function to calculate the total size of a directory
def calculate_directory_size(directory: str) -> int:
    """
    Calculate the total size of all files within a directory, including subdirectories.

    Args:
        directory (str): The path to the directory.

    Returns:
        int: Total size in bytes.
    """
    # Initialize the total size counter
    total_size = 0
    # Traverse the directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            # Add the size of each file to the total size
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size

# Function to generate a manifest entry for specific file types
def generate_manifest_entry(directory: str, file_types: List[str]) -> Dict:
    """
    Generate a manifest entry for a directory, focusing on specific file types, including `.zarr`.

    Args:
        directory (str): Path to the directory to generate a manifest entry for.
        file_types (List[str]): List of file extensions to include in the manifest.

    Returns:
        Dict: A dictionary representing the manifest entry for the directory.
    """
    # Initialize the manifest entry dictionary
    manifest_entry = {}
    # Traverse the directory tree recursively
    for root, dirs, files in os.walk(directory):
        # Process files with specified extensions
        for file in files:
            if any(file.endswith(ext) for ext in file_types):
                # Construct the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Get the size of the file
                file_size = os.path.getsize(os.path.join(root, file))
                # Add the file entry to the manifest
                manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": file_size}

        # Process directories that are `.zarr`
        for dir_name in dirs:
            if dir_name.endswith(".zarr"):
                # Construct the `.zarr` path and its relative path
                zarr_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(zarr_path, directory)
                # Calculate the total size of the `.zarr` directory
                zarr_size = calculate_directory_size(zarr_path)
                # Add the `.zarr` directory entry to the manifest
                manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": zarr_size}

        # Prevent descending further into `.zarr` directories
        dirs[:] = [d for d in dirs if not d.endswith(".zarr")]

    return manifest_entry

# Function to update the manifest
def update_manifest(config: Dict) -> None:
    """
    Update the manifest file based on the current state of the directories defined in the config.

    Args:
        config (Dict): The configuration dictionary containing paths to the directories.
    """
    # Define the file types to include in the manifest
    file_types = [".tif", ".csv", ".h5ad"]

    # Load the existing manifest file
    manifest = load_manifest()

    # Iterate over the Crunch configurations in the config
    for crunch_name, crunch_config in config["crunches"].items():
        # Get the project directory for the current Crunch
        project_dir = crunch_config["paths"]["project_dir"]

        # Check if the project directory exists
        if os.path.exists(project_dir):
            print(f"[INFO] Updating manifest for {crunch_name}...")
            # Generate and update the manifest entry for the current Crunch
            manifest[crunch_name] = generate_manifest_entry(project_dir, file_types)
        else:
            # Warn if the directory does not exist
            print(f"[WARNING] Directory not found for {crunch_name}: {project_dir}")

    # Save the updated manifest to the manifest file
    save_manifest(manifest)
    print("[INFO] Manifest updated successfully.")

# Main function to generate the manifest
def main():
    """
    Main function to update the manifest based on the current directory structure.

    Steps:
    1. Load the config.yaml file.
    2. Generate or update the manifest.json file.
    """
    import yaml  # Import YAML library for config loading

    try:
        # Open and load the configuration file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Update the manifest based on the loaded configuration
        update_manifest(config)

    except FileNotFoundError as e:
        # Handle file not found error
        print(f"[ERROR] {e}")
    except yaml.YAMLError as e:
        # Handle YAML parsing error
        print(f"[ERROR] Failed to parse YAML file: {e}")
    except Exception as e:
        # Handle any unexpected errors
        print(f"[ERROR] An unexpected error occurred: {e}")

# Entry point for the script
if __name__ == "__main__":
    main()
