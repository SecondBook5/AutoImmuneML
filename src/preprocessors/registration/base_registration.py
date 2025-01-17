# File: src/preprocessors/registration/base_registration.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class BaseRegistration(ABC):
    """
    Abstract base class for image registration.
    Provides common utilities and defines the interface for specific registration methods.
    """

    def __init__(self):
        """
        Initialize the base class and common parameters for registration.
        """
        # Define error metrics as a dictionary mapping metric names to their respective functions
        self.error_metrics = {'MSE': self.mean_squared_error, 'SSIM': self.structural_similarity}

    @staticmethod
    def preprocess_image(image: np.ndarray, to_gray: bool = True) -> np.ndarray:
        """
        Preprocess an image by normalizing and optionally converting it to grayscale.

        Args:
            image (np.ndarray): Input image to preprocess.
            to_gray (bool): Whether to convert the image to grayscale.

        Returns:
            np.ndarray: Preprocessed image.
        """
        # If the image has three channels (color image) and to_gray is True, convert to grayscale
        if to_gray and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize the image values to the range [0, 1] for consistency in processing
        return cv2.normalize(image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    @staticmethod
    def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two images.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).

        Returns:
            float: Mean Squared Error value.
        """
        # Compute the squared difference between the two images and take the mean
        return np.mean((image1 - image2) ** 2)

    @staticmethod
    def structural_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Structural Similarity Index (SSIM) between two images.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).

        Returns:
            float: SSIM value, indicating structural similarity.
        """
        # Use the SSIM function from skimage to compute the similarity between the images
        return ssim(image1, image2, data_range=image2.max() - image2.min())

    @abstractmethod
    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to register two images. Must be implemented by subclasses.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image to align.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Registered target image and transformation matrix.
        """
        # Subclasses are required to define this method to implement specific registration logic
        pass

    def compute_error(self, image1: np.ndarray, image2: np.ndarray, metric: str = 'MSE') -> float:
        """
        Compute the alignment error between two images using the specified metric.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).
            metric (str): Error metric to use ('MSE' or 'SSIM').

        Returns:
            float: Computed error value.

        Raises:
            ValueError: If an unsupported metric is provided.
        """
        # Validate if the specified metric is supported, raise an error otherwise
        if metric not in self.error_metrics:
            raise ValueError(f"Unsupported metric: {metric}. Choose from {list(self.error_metrics.keys())}.")
        # Compute and return the error using the specified metric
        return self.error_metrics[metric](image1, image2)

    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> None:
        """
        Save an image to a specified file path.

        Args:
            image (np.ndarray): Image to save.
            file_path (str): File path where the image will be saved.

        Raises:
            IOError: If the image cannot be saved.
        """
        # Attempt to save the image to the specified file path
        try:
            success = cv2.imwrite(file_path, (image * 255).astype(np.uint8))
            if not success:
                raise IOError(f"Failed to save image to {file_path}.")
        except Exception as e:
            # Raise an error if the image cannot be saved due to any exception
            raise IOError(f"Error while saving image to {file_path}: {str(e)}")
