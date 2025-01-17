# File: src/validators/image_validator.py
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple


class ImageValidator:
    """
    Validates image registration quality using quantitative metrics.

    Supports SSIM, Dice coefficient, and pixel-wise error to assess alignment quality.
    """

    @staticmethod
    def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two images.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (registered).

        Returns:
            float: SSIM value, where 1.0 indicates perfect similarity.
        """
        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions.")

        # Compute SSIM
        ssim_value = ssim(image1, image2, data_range=image2.max() - image2.min())
        return ssim_value

    @staticmethod
    def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Dice coefficient between two binary masks.

        Args:
            mask1 (np.ndarray): First binary mask.
            mask2 (np.ndarray): Second binary mask.

        Returns:
            float: Dice coefficient, where 1.0 indicates perfect overlap.
        """
        # Ensure masks are binary
        if not (np.array_equal(mask1, mask1.astype(bool)) and np.array_equal(mask2, mask2.astype(bool))):
            raise ValueError("Input masks must be binary (0 or 1).")

        # Compute intersection and union
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2)

        # Compute Dice coefficient
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  # Handle edge cases
        return 2 * intersection / union

    @staticmethod
    def compute_pixel_error(image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute pixel-wise error between two images.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (registered).

        Returns:
            float: Mean squared pixel error between the images.
        """
        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same dimensions.")

        # Compute Mean Squared Error (MSE)
        mse = np.mean((image1 - image2) ** 2)
        return mse

    @staticmethod
    def validate_registration(
        image1: np.ndarray, image2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
    ) -> dict:
        """
        Validate registration quality using multiple metrics.

        Args:
            image1 (np.ndarray): First image (reference).
            image2 (np.ndarray): Second image (registered).
            mask1 (np.ndarray): First binary mask (reference segmentation).
            mask2 (np.ndarray): Second binary mask (registered segmentation).

        Returns:
            dict: Dictionary containing validation metrics (SSIM, Dice, Pixel Error).
        """
        metrics = {}

        # Compute SSIM
        metrics["SSIM"] = ImageValidator.compute_ssim(image1, image2)

        # Compute Dice coefficient
        metrics["Dice"] = ImageValidator.compute_dice(mask1, mask2)

        # Compute Pixel Error
        metrics["Pixel Error"] = ImageValidator.compute_pixel_error(image1, image2)

        return metrics
