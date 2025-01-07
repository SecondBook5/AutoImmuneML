# File: src/preprocessors/base_preprocessor.py
from abc import ABC, abstractmethod
import os

class BasePreprocessor(ABC):
    """
    Abstract base class for preprocessors to ensure a consistent interface.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the base preprocessor.

        Args:
            output_dir (str): Directory to save preprocessed data.
        """
        self.output_dir = output_dir

    @abstractmethod
    def preprocess(self, data, **kwargs):
        """
        Abstract method to preprocess data. Must be implemented by subclasses.

        Args:
            data: The input data to preprocess.
            kwargs: Additional arguments specific to the preprocessor.
        """
        pass

    def save(self, data, filename: str):
        """
        Save preprocessed data to the specified output directory.

        Args:
            data: Preprocessed data to save.
            filename (str): Name of the file to save.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        data.save(filepath)
        print(f"Saved preprocessed data to {filepath}")

    def validate(self, data):
        """
        Optional validation method to check data integrity.
        """
        pass
