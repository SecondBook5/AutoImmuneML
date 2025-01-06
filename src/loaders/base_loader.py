# File: src/loaders/base_loader.py
import os
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """
    Abstract base class for file loaders, providing common utilities for path validation.
    """

    def __init__(self, path: str):
        """
        Initialize the BaseLoader with a file path.

        Args:
            path (str): Path to the file or directory to load.
        """
        # Store the file path
        self.path = path
        # Validate that the path exists
        self._validate_path()

    def _validate_path(self):
        """
        Ensure the provided path exists.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        # Check if the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path not found: {self.path}")

    @abstractmethod
    def load(self):
        """
        Abstract method to load the file. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass
