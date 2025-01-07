# File: src/downloader/token_handler.py
import os


class TokenHandler:
    """
    A utility class to handle CrunchDAO authentication tokens.

    This class provides methods to retrieve and validate tokens stored in a file.
    """

    def __init__(self, token_file: str):
        """
        Initialize the TokenHandler with the path to the token file.

        Args:
            token_file (str): Path to the file containing authentication tokens.
        """
        self.token_file = token_file

    def get_token(self, line_number: int) -> str:
        """
        Retrieve the authentication token from the specified line in the token file.

        Args:
            line_number (int): The line number (1-based) of the token to retrieve.

        Returns:
            str: The authentication token.

        Raises:
            FileNotFoundError: If the token file does not exist.
            ValueError: If the specified line number is invalid.
        """
        # Check if the token file exists
        if not os.path.exists(self.token_file):
            raise FileNotFoundError(f"Token file not found: {self.token_file}")

        # Read the token file line by line
        with open(self.token_file, "r") as file:
            lines = file.readlines()

        # Validate the requested line number
        if line_number < 1 or line_number > len(lines):
            raise ValueError(f"Invalid line number {line_number}. Token file has {len(lines)} lines.")

        # Return the token from the specified line, stripping any trailing whitespace
        return lines[line_number - 1].strip()

    def validate_token(self, token: str) -> bool:
        """
        Validate the format of an authentication token.

        Args:
            token (str): The token to validate.

        Returns:
            bool: True if the token format is valid, False otherwise.
        """
        # Example validation: Ensure the token is non-empty and has a minimum length
        return bool(token) and len(token) > 10
