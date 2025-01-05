# File: src/validators/validation_logger.py
from typing import Dict, List  # Import type hints for improved code clarity

class ValidationLogger:
    """
    A class to log validation results and errors during the spatial data validation process.

    This logger is designed to record the validation status (valid, invalid, or error) and
    track any error messages for each file being validated.
    """

    def __init__(self):
        """
        Initialize the ValidationLogger with empty result and error dictionaries.
        """
        # Dictionary to store validation results for each file
        self.results: Dict[str, str] = {}
        # Dictionary to store error messages for files that encountered validation issues
        self.errors: Dict[str, str] = {}

    def record_result(self, file_path: str, status: str):
        """
        Record the validation result for a specific file.

        Args:
            file_path (str): The path to the file being validated.
            status (str): The validation status (e.g., "valid", "invalid").
        """
        # Store the validation status for the given file path
        self.results[file_path] = status

    def record_error(self, file_path: str, error_message: str):
        """
        Record an error message for a specific file.

        Args:
            file_path (str): The path to the file that encountered an error.
            error_message (str): The error message describing the issue.
        """
        # Store the error message for the given file path
        self.errors[file_path] = error_message

    def get_results(self) -> Dict[str, str]:
        """
        Retrieve all recorded validation results.

        Returns:
            Dict[str, str]: A dictionary containing file paths and their validation statuses.
        """
        # Return the dictionary of validation results
        return self.results

    def get_errors(self) -> Dict[str, str]:
        """
        Retrieve all recorded error messages.

        Returns:
            Dict[str, str]: A dictionary containing file paths and their error messages.
        """
        # Return the dictionary of error messages
        return self.errors

    def summarize(self) -> Dict[str, int]:
        """
        Summarize the validation results, providing counts of valid, invalid, and error statuses.

        Returns:
            Dict[str, int]: A dictionary summarizing the counts of each validation outcome.
        """
        # Count the number of valid files
        valid_count = sum(1 for status in self.results.values() if status == "valid")
        # Count the number of invalid files
        invalid_count = sum(1 for status in self.results.values() if status == "invalid")
        # Count the number of files with errors
        error_count = len(self.errors)

        # Return a summary of the counts
        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "errors": error_count,
        }

    def print_summary(self):
        """
        Print a summary of the validation results to the console.
        """
        # Retrieve the summary of validation results
        summary = self.summarize()
        # Print the counts of valid, invalid, and error statuses
        print("\nValidation Summary:")
        print(f"Valid files: {summary['valid']}")
        print(f"Invalid files: {summary['invalid']}")
        print(f"Files with errors: {summary['errors']}")

    def print_errors(self):
        """
        Print all error messages to the console.
        """
        # Check if there are any errors to display
        if not self.errors:
            print("\nNo errors recorded.")
            return

        # Print each file path and its associated error message
        print("\nValidation Errors:")
        for file_path, error_message in self.errors.items():
            print(f"File: {file_path}\nError: {error_message}\n")
