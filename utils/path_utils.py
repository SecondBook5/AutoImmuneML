# File: utils/path_utils.py
# Utility functions for resolving, validating, and managing paths dynamically.

import os
import re
from typing import Dict

# Function to resolve placeholders in a path string with actual values from a dictionary
def resolve_path(path: str, variables: Dict[str, str]) -> str:
    """
    Resolve placeholders in a path string using the provided variables.

    Args:
        path (str): Path string with placeholders (e.g., {base_dir}/data).
        variables (Dict[str, str]): Dictionary of variables to replace placeholders.

    Returns:
        str: Resolved path with placeholders replaced by actual values.

    Raises:
        ValueError: If unresolved placeholders remain in the path.

    How It Works:
        - Placeholders enclosed in curly braces (e.g., {base_dir}) are replaced
          with corresponding values from the variables dictionary.
        - If placeholders remain unresolved, a ValueError is raised.
    """
    # Return the path unchanged if it is None or empty
    if not path:
        return path

    try:
        # Replace each placeholder in the path with its corresponding value
        for key, value in variables.items():
            path = path.replace(f"{{{key}}}", value)

        # Extract unresolved placeholders, if any, and raise an error
        unresolved = extract_placeholders(path)
        if unresolved:
            raise ValueError(f"Unresolved placeholders in path '{path}': {unresolved}")

        # Return the resolved path
        return path
    except Exception as e:
        # Raise a ValueError if an error occurs during path resolution
        raise ValueError(f"Error resolving path '{path}': {e}")

def resolve_path_recursively(path: str, variables: Dict[str, str]) -> str:
    """
    Resolve placeholders in a path string recursively using the provided variables.

    Args:
        path (str): Path string with placeholders (e.g., {base_dir}/data).
        variables (Dict[str, str]): Dictionary of variables to replace placeholders.

    Returns:
        str: Resolved path with placeholders replaced by actual values.

    Raises:
        ValueError: If unresolved placeholders remain after recursive resolution.

    How It Works:
        - Keeps resolving placeholders until none are left in the path.
        - Uses `resolve_path` in a loop to handle nested placeholders.
    """
    # Keep resolving placeholders until none are left
    while "{" in path and "}" in path:
        path = resolve_path(path, variables)
    return path

def extract_placeholders(path: str) -> list:
    """
    Extract unresolved placeholders from a path string.

    Args:
        path (str): Path string to analyze.

    Returns:
        list: List of placeholders found in the path (e.g., ['base_dir']).

    How It Works:
        - Uses a regular expression to find all substrings enclosed in curly braces.
        - Returns a list of placeholder names.
    """
    # Use regex to find all placeholders enclosed in curly braces
    return re.findall(r"{(.*?)}", path)

def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path (str): Path to the directory.

    Raises:
        RuntimeError: If the directory cannot be created.

    How It Works:
        - Checks if the directory exists.
        - Creates the directory (and any missing parent directories) if it does not exist.
    """
    try:
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Print success message for debugging purposes
        print(f"[\u2714] Directory ensured: {path}")
    except OSError as e:
        # Raise a RuntimeError if directory creation fails
        raise RuntimeError(f"Error ensuring directory '{path}': {e}")

# Function to normalize a path for consistent formatting
def normalize_path(path: str) -> str:
    """
    Normalize a path to ensure consistent formatting.

    Args:
        path (str): Path to normalize.

    Returns:
        str: Normalized absolute path.

    How It Works:
        - Converts the path to an absolute path.
        - Normalizes the path to remove redundant separators or up-level references.
    """
    # Normalize and return the absolute path
    return os.path.normpath(os.path.abspath(path))

def get_global_variables(config: Dict) -> Dict[str, str]:
    """
    Extract global variables from the configuration dictionary.

    Args:
        config (Dict): Parsed configuration dictionary.

    Returns:
        Dict[str, str]: Dictionary of global variables for path resolution.

    How It Works:
        - Extracts the 'global' section from the configuration.
        - Returns a dictionary containing base_dir and data_dir values.
    """
    # Retrieve the global section from the configuration
    global_config = config.get("global", {})

    # Extract and return global variables relevant for path resolution
    return {
        "base_dir": global_config.get("base_dir", ""),
        "data_dir": global_config.get("data_dir", ""),
    }

# Function to merge global variables with Crunch-specific variables
def merge_variables(global_vars: Dict[str, str], crunch_vars: Dict[str, str]) -> Dict[str, str]:
    """
    Merge global variables with Crunch-specific variables.

    Args:
        global_vars (Dict[str, str]): Global variables for path resolution.
        crunch_vars (Dict[str, str]): Crunch-specific variables for path resolution.

    Returns:
        Dict[str, str]: Merged dictionary of variables.

    How It Works:
        - Combines the global and Crunch-specific variables.
        - Crunch-specific variables take precedence over global variables.
    """
    # Merge global and crunch-specific variables, prioritizing crunch-specific
    merged = {**global_vars, **crunch_vars}

    # Print debug information about the merged variables
    print(f"[DEBUG] Merged Variables: {merged}")

    # Return the merged dictionary
    return merged

def resolve_all_paths(config: Dict) -> Dict:
    """
    Resolve all paths in the configuration dynamically and recursively.

    Args:
        config (Dict): Parsed configuration dictionary with placeholders.

    Returns:
        Dict: Configuration with all paths resolved.

    Raises:
        ValueError: If placeholders remain unresolved after resolution.

    How It Works:
        - Resolves global paths using global variables.
        - Resolves Crunch-specific paths by merging global and Crunch-specific variables.
        - Ensures no unresolved placeholders remain in the final configuration.
    """
    # Extract global variables from the configuration
    global_vars = get_global_variables(config)

    # Create a copy of the configuration to resolve paths without modifying the original
    resolved_config = config.copy()

    # Resolve global paths recursively
    for key, value in resolved_config["global"].items():
        if isinstance(value, str):
            resolved_config["global"][key] = resolve_path_recursively(value, global_vars)

    # Resolve Crunch-specific paths recursively
    for crunch_name, crunch_config in resolved_config["crunches"].items():
        # Extract Crunch-specific variables
        crunch_vars = {"project_dir": crunch_config["paths"].get("project_dir")}

        # Merge global and Crunch-specific variables
        all_vars = merge_variables(global_vars, crunch_vars)

        # Resolve each path in the Crunch's paths dictionary recursively
        for path_key, path_value in crunch_config["paths"].items():
            resolved_config["crunches"][crunch_name]["paths"][path_key] = resolve_path_recursively(path_value, all_vars)

    # Return the fully resolved configuration
    return resolved_config
