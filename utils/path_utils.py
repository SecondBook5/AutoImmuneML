# File: utils/path_utils.py
# This file contains utility functions for resolving and managing paths with placeholders in a configuration file.
from typing import Dict
import os
import re

# Function to resolve placeholders in a path string with actual values from a dictionary
def resolve_path(path: str, variables: Dict[str, str]) -> str:
    """
    Resolve a path with placeholders using the provided variables.

    Args:
        path (str): Path string with placeholders (e.g., {base_dir}).
        variables (Dict[str, str]): Dictionary of variables to replace in the path.

    Returns:
        str: Resolved path with placeholders replaced by actual values.

    Raises:
        ValueError: If path resolution fails due to missing or invalid variables.
    """
    # Check if the path is None or empty and return it unchanged if so
    if not path:
        return path
    try:
        # Replace all placeholders in the path with corresponding values from the variables dictionary
        for key, value in variables.items():
            path = path.replace(f"{{{key}}}", value)
        # Return the fully resolved path
        return path
    except Exception as e:
        # Raise a ValueError if an error occurs during path resolution
        raise ValueError(f"Error resolving path '{path}': {e}")

# Function to resolve placeholders with fallback to default values
def resolve_path_with_defaults(path: str, variables: Dict[str, str], defaults: Dict[str, str] = None) -> str:
    """
    Resolve a path with placeholders using the provided variables and fallback defaults.

    Args:
        path (str): Path string with placeholders (e.g., {base_dir}).
        variables (Dict[str, str]): Dictionary of variables to replace in the path.
        defaults (Dict[str, str]): Dictionary of default values for placeholders.

    Returns:
        str: Resolved path with placeholders replaced by actual values or defaults.
    """
    # Return the path unchanged if it is None or empty
    if not path:
        return path
    # Initialize defaults if not provided
    defaults = defaults or {}
    try:
        # Iterate through both variables and defaults to replace placeholders
        for key in set(variables) | set(defaults):
            value = variables.get(key, defaults.get(key, ""))
            path = path.replace(f"{{{key}}}", value)
        # Return the resolved path
        return path
    except Exception as e:
        # Raise a ValueError if an error occurs during resolution
        raise ValueError(f"Error resolving path '{path}' with defaults: {e}")

# Function to validate that a resolved path contains no placeholders and is valid
def validate_resolved_path(path: str) -> bool:
    """
    Validate that a resolved path contains no placeholders and is valid.

    Args:
        path (str): Resolved path to validate.

    Returns:
        bool: True if the path is valid, False otherwise.

    Raises:
        ValueError: If the path still contains unresolved placeholders.
    """
    # Check for unresolved placeholders in the path
    if "{" in path or "}" in path:
        raise ValueError(f"Path '{path}' contains unresolved placeholders.")
    # Check if the directory portion of the path exists
    if not os.path.exists(os.path.dirname(path)):
        print(f"[✘] Directory does not exist for path: {path}")
        return False
    # Print success message and return True if path is valid
    print(f"[✔] Path is valid: {path}")
    return True

# Function to ensure a directory exists, creating it if necessary
def ensure_directory_exists(path: str):
    """
    Ensure that a directory exists; create it if it does not.

    Args:
        path (str): Path to the directory to check or create.

    Returns:
        str: The absolute path of the ensured directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)
        # Print success message and return the absolute path of the directory
        print(f"[✔] Directory ensured: {path}")
        return os.path.abspath(path)
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
    """
    # Normalize and return the absolute path
    return os.path.normpath(os.path.abspath(path))

# Function to extract unresolved placeholders from a path
def extract_placeholders(path: str) -> list:
    """
    Extract placeholders (e.g., {base_dir}) from a path.

    Args:
        path (str): Path string to analyze.

    Returns:
        list: List of placeholders found in the path.
    """
    # Use a regular expression to find all placeholders in the path
    return re.findall(r"{(.*?)}", path)

# Function to extract global variables from the configuration dictionary
def get_global_variables(config: Dict) -> Dict[str, str]:
    """
    Extract global variables from the configuration.

    Args:
        config (Dict): Parsed configuration dictionary.

    Returns:
        Dict[str, str]: Dictionary of global variables for path resolution.
    """
    # Retrieve the global section from the configuration file
    global_config = config.get("global", {})
    # Return a dictionary containing only relevant global variables for path resolution
    return {
        "base_dir": global_config.get("base_dir", ""),
        "data_dir": global_config.get("data_dir", ""),
    }

# Function to merge global variables with Crunch-specific variables
def merge_variables(global_vars: Dict[str, str], crunch_vars: Dict[str, str]) -> Dict[str, str]:
    """
    Merge global and crunch-specific variables.

    Args:
        global_vars (Dict[str, str]): Global variables for path resolution.
        crunch_vars (Dict[str, str]): Crunch-specific variables for path resolution.

    Returns:
        Dict[str, str]: Merged dictionary of variables.
    """
    # Merge the global and crunch-specific dictionaries, with Crunch-specific variables overriding global ones
    return {**global_vars, **crunch_vars}
