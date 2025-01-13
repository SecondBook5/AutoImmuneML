# File: src/validators/zarr_validator.py
import os  # For file and directory operations
import spatialdata as sd  # For validating SpatialData structures
from typing import List, Dict, Union  # For type annotations

class ZARRValidator:
    """
    Validates Zarr datasets and directories containing Zarr datasets.

    Features:
    - Checks the existence of `.zarr` directories.
    - Validates the structure of Zarr datasets.
    - Reports missing keys, invalid files, or structural inconsistencies.
    """

    def __init__(self, path: str):
        """
        Initialize the ZARRValidator with the path to a Zarr dataset or directory.

        Args:
            path (str): Path to a single Zarr directory or a directory containing Zarr datasets.
        """
        # Save the provided path to an instance variable
        self.path = path

    def validate(self) -> Dict[str, Union[str, List[str]]]:
        """
        Validate the specified path.

        - If the path points to a single `.zarr` directory, validate its structure.
        - If the path points to a directory, validate all `.zarr` directories in the directory.

        Returns:
            Dict[str, Union[str, List[str]]]:
                - "status": Overall validation status (e.g., "valid", "invalid").
                - "errors": List of errors found during validation.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        # Ensure the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        # Initialize an error log
        errors = []

        # Validate a single `.zarr` directory or a directory containing `.zarr` directories
        if os.path.isdir(self.path) and self.path.endswith(".zarr"):
            # Validate the single Zarr directory
            is_valid, error = self._validate_single_zarr(self.path)
            if not is_valid:
                errors.append(error)
        elif os.path.isdir(self.path):
            # Validate all `.zarr` directories in the directory
            zarr_dirs = [
                os.path.join(self.path, item)
                for item in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, item)) and item.endswith(".zarr")
            ]

            if not zarr_dirs:
                errors.append(f"No valid .zarr datasets found in directory: {self.path}")
            else:
                for zarr_dir in zarr_dirs:
                    is_valid, error = self._validate_single_zarr(zarr_dir)
                    if not is_valid:
                        errors.append(error)
        else:
            # Raise an error if the path is neither a `.zarr` directory nor a directory containing `.zarr` directories
            raise ValueError(f"Invalid path: {self.path}. Must be a .zarr directory or a directory containing .zarr directories.")

        # Return validation results
        return {
            "status": "valid" if not errors else "invalid",
            "errors": errors,
        }

    def _validate_single_zarr(self, zarr_path: str) -> (bool, str):
        """
        Validate a single Zarr dataset.

        Args:
            zarr_path (str): Path to the Zarr dataset.

        Returns:
            (bool, str): Tuple containing a boolean (valid or not) and an error message (if any).
        """
        try:
            # Attempt to read the Zarr dataset using SpatialData
            sd.read_zarr(zarr_path)
            return True, ""  # Return valid status if no exceptions occur
        except Exception as e:
            # Catch exceptions and return them as errors
            return False, f"Error validating {zarr_path}: {str(e)}"

    def report(self) -> None:
        """
        Generate and print a validation report for the specified path.
        """
        # Perform validation
        results = self.validate()

        # Print the validation status
        print(f"Validation Status: {results['status']}")

        # Print errors if any
        if results["errors"]:
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")

# Example Usage
if __name__ == "__main__":
    # Define the paths to the Zarr datasets
    zarr_paths = [
        "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1/data/UC1_I.zarr",
        "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1/data",
        "/mnt/d/AutoImmuneML/broad-1-autoimmune-crunch1/data/UC1_NI.zarr",
    ]

    # Loop through each path and validate
    for zarr_path in zarr_paths:
        print(f"[INFO] Validating Zarr dataset: {zarr_path}")

        # Create an instance of ZARRValidator for the given path
        validator = ZARRValidator(zarr_path)

        # Generate a report for the Zarr dataset
        validator.report()
        print("-" * 50)  # Separator for readability

