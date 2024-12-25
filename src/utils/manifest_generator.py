# File: src/utils/manifest_generator.py
import os  # Module for file and directory handling
import json  # Module for handling JSON data
from typing import Dict, List  # Type hints for better clarity and robustness


# Define the default path to the manifest file
MANIFEST_FILE = "../../manifest.json"


def load_manifest() -> Dict:
    """
    Load the manifest file if it exists; otherwise, return an empty dictionary.

    Returns:
        Dict: The current manifest data or an empty dictionary if no manifest exists.
    """
    # Check if the manifest file exists
    if os.path.exists(MANIFEST_FILE):
        # Open and load the manifest JSON file
        with open(MANIFEST_FILE, "r") as file:
            return json.load(file)  # Parse and return the JSON data as a dictionary
    return {}  # Return an empty dictionary if the file does not exist


def save_manifest(manifest: Dict) -> None:
    """
    Save the manifest data to the manifest file.

    Args:
        manifest (Dict): The manifest data to save.
    """
    # Write the dictionary to the manifest file in JSON format
    with open(MANIFEST_FILE, "w") as file:
        json.dump(manifest, file, indent=4)  # Use indentation for readability


def calculate_directory_size(directory: str) -> int:
    """
    Calculate the total size of all files in a directory, including subdirectories.

    Args:
        directory (str): The directory to calculate size for.

    Returns:
        int: Total size in bytes.
    """
    total_size = 0  # Initialize the total size counter
    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            # Add the size of each file to the total size
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size


def generate_manifest_entry(directory: str, file_types: List[str]) -> Dict:
    """
    Generate a manifest entry for a directory, focusing on specified file types.

    Args:
        directory (str): Directory to scan for files.
        file_types (List[str]): List of file extensions to include (e.g., ".tif", ".zarr").

    Returns:
        Dict: A dictionary representing the manifest entry.
    """
    manifest_entry = {}  # Initialize the manifest entry dictionary

    # Walk through the directory and process files and directories
    for root, dirs, files in os.walk(directory):
        # Add files matching the specified extensions
        for file in files:
            if any(file.endswith(ext) for ext in file_types):
                relative_path = os.path.relpath(os.path.join(root, file), directory)  # Relative file path
                file_size = os.path.getsize(os.path.join(root, file))  # File size in bytes
                manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": file_size}  # Add to manifest

        # Handle `.zarr` directories explicitly
        for dir_name in dirs:
            if dir_name.endswith(".zarr"):
                zarr_path = os.path.join(root, dir_name)  # Full `.zarr` directory path
                relative_path = os.path.relpath(zarr_path, directory)  # Relative `.zarr` directory path
                zarr_size = calculate_directory_size(zarr_path)  # Calculate the size of the `.zarr` directory
                manifest_entry[relative_path] = {"status": "downloaded", "size_bytes": zarr_size}  # Add to manifest

        # Prevent descending further into `.zarr` directories
        dirs[:] = [d for d in dirs if not d.endswith(".zarr")]

    return manifest_entry


def update_manifest(config: Dict) -> None:
    """
    Update the manifest file based on the state of directories defined in the config.

    Args:
        config (Dict): Configuration dictionary with paths to directories.
    """
    file_types = [".tif", ".csv", ".h5ad"]  # Define the file extensions to track
    manifest = load_manifest()  # Load the existing manifest

    # Process each Crunch configuration in the config
    for crunch_name, crunch_config in config["crunches"].items():
        project_dir = crunch_config["paths"]["project_dir"]  # Get the project directory path

        # Check if the project directory exists
        if os.path.exists(project_dir):
            print(f"Updating manifest for {crunch_name}...")  # Notify the user about the update
            manifest[crunch_name] = generate_manifest_entry(project_dir, file_types)  # Generate the manifest entry
        else:
            print(f"Directory not found for {crunch_name}: {project_dir}")  # Warn if the directory is missing

    save_manifest(manifest)  # Save the updated manifest to file
    print("Manifest updated successfully.")  # Confirm the update


def main():
    """
    Main function to load the config and update the manifest.
    """
    import yaml  # Import YAML for config handling

    try:
        # Load the configuration file
        with open("../../config.yaml", "r") as file:
            config = yaml.safe_load(file)  # Parse the YAML file into a dictionary

        # Update the manifest based on the loaded configuration
        update_manifest(config)

    except FileNotFoundError as e:
        # Handle missing files gracefully
        print(f"Error: {e}")
    except yaml.YAMLError as e:
        # Handle YAML parsing errors
        print(f"Failed to parse YAML file: {e}")
    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


# Entry point for the script
if __name__ == "__main__":
    main()
