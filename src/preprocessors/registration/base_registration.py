# File: src/preprocessors/registration/base_registration.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class BaseRegistration(ABC):
    """
    Abstract base class for image registration.

    This class provides a reusable foundation for implementing specific registration methods,
    such as affine or deformable registration. It includes utilities for preprocessing images,
    evaluating registration quality using multiple error metrics, and saving outputs.

    **Concepts**:
    - **Error Metrics**: Used to evaluate the similarity between two images post-registration.
        Examples include Mean Squared Error (MSE), Structural Similarity Index (SSIM), and Dice Coefficient.
    - **Abstract Methods**: Define required functionality for subclasses, ensuring flexibility for specific use cases.
    """

    def __init__(self):
        """
        Initialize the base class with error metrics.

        Provides a dictionary of error metrics, mapping metric names to their respective functions.
        """
        # Dictionary to store error metrics: metric name -> function
        self.error_metrics = {
            'MSE': self.mean_squared_error,         # Mean Squared Error
            'SSIM': self.structural_similarity,    # Structural Similarity Index
            'Dice': self.dice_coefficient,         # Dice Coefficient for binary masks
            'PSNR': self.peak_signal_to_noise_ratio  # Peak Signal-to-Noise Ratio
        }

    @staticmethod
    def preprocess_image(image: np.ndarray, to_gray: bool = True) -> np.ndarray:
        """
        Preprocess an image by normalizing and optionally converting it to grayscale.

        **Concept**: Preprocessing ensures that images are in a consistent format for registration.

        Args:
            image (np.ndarray): Input image to preprocess.
            to_gray (bool): Whether to convert the image to grayscale.

        Returns:
            np.ndarray: Preprocessed image.
        """
        # Check if the image is color (3 channels) and convert to grayscale if required
        if to_gray and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize the pixel values to the range [0, 1]
        return cv2.normalize(image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    @staticmethod
    def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two images.

        **Concept**: MSE measures pixel-wise differences between two images.
        A lower value indicates better alignment.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).

        Returns:
            float: Mean Squared Error value.
        """
        # Compute the squared differences between images and take the mean
        return np.mean((image1 - image2) ** 2)

    @staticmethod
    def structural_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Structural Similarity Index (SSIM) between two images.

        **Concept**: SSIM evaluates perceptual similarity by comparing luminance, contrast, and structure.
        A value closer to 1 indicates higher similarity.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).

        Returns:
            float: SSIM value.
        """
        # Compute SSIM using skimage
        return ssim(image1, image2, data_range=image2.max() - image2.min())

    @staticmethod
    def dice_coefficient(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Dice Coefficient between two binary masks.

        **Concept**: The Dice Coefficient measures the overlap between two sets.
        It ranges from 0 (no overlap) to 1 (perfect overlap).

        Args:
            image1 (np.ndarray): First binary mask.
            image2 (np.ndarray): Second binary mask.

        Returns:
            float: Dice Coefficient value.
        """
        # Calculate the intersection and union of the binary masks
        intersection = np.sum(image1 * image2)
        union = np.sum(image1) + np.sum(image2)
        # Compute Dice Coefficient, handling edge cases where the union is zero
        return 2 * intersection / union if union > 0 else 1.0

    @staticmethod
    def peak_signal_to_noise_ratio(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

        **Concept**: PSNR compares the peak signal power to the noise power.
        Higher values indicate better quality alignment.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (target).

        Returns:
            float: PSNR value in decibels (dB).
        """
        # Compute the Mean Squared Error
        mse = np.mean((image1 - image2) ** 2)
        # Handle edge case where MSE is zero (perfect similarity)
        if mse == 0:
            return float('inf')
        # Compute PSNR using the formula: 20 * log10(MAX_I / sqrt(MSE))
        max_pixel = 1.0  # Assuming normalized images
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    @abstractmethod
    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to register two images. Must be implemented by subclasses.

        **Concept**: Subclasses define the specific logic for aligning `image2` with `image1`.

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
            metric (str): Error metric to use ('MSE', 'SSIM', 'Dice', 'PSNR').

        Returns:
            float: Computed error value.

        Raises:
            ValueError: If an unsupported metric is provided.
        """
        # Check if the metric is supported; raise an error otherwise
        if metric not in self.error_metrics:
            raise ValueError(f"Unsupported metric: {metric}. Choose from {list(self.error_metrics.keys())}.")
        # Compute the error using the specified metric
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
            # Attempt to save the image as an 8-bit image
            success = cv2.imwrite(file_path, (image * 255).astype(np.uint8))
            if not success:
                raise IOError(f"Failed to save image to {file_path}.")
        except Exception as e:
            # Raise an error if saving fails
            raise IOError(f"Error while saving image to {file_path}: {str(e)}")
