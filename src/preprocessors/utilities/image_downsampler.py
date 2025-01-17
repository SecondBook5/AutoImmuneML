# File: src/preprocessors/registration/image_downsampler.py
"""
Image Downsampler Module

This module is responsible for downsampling large images while maintaining image quality.
It is designed to process multi-channel images efficiently, using anti-aliasing to reduce
artifacts during resizing.

The module is part of the image registration pipeline and can be used as a preprocessing
step for visualization or analysis.

Author: Your Name
"""

import numpy as np
from skimage.transform import resize


class ImageDownsampler:
    """
    Handles the downsampling of images with support for multi-channel images.

    This class provides a method to downsample images based on a specified factor,
    ensuring the resized image retains essential details while reducing computational costs.
    """

    def __init__(self, downsample_factor: int = 4):
        """
        Initialize the ImageDownsampler with a specified downsample factor.

        Args:
            downsample_factor (int): Factor by which the image dimensions will be reduced.
        """
        if downsample_factor <= 0:
            raise ValueError("Downsample factor must be greater than zero.")
        self.downsample_factor = downsample_factor

    def downsample(self, image: np.ndarray) -> np.ndarray:
        """
        Downsample the given image by the configured factor.

        Args:
            image (np.ndarray): Input image to downsample. Can be grayscale or multi-channel.

        Returns:
            np.ndarray: Downsampled image.
        """
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if image.ndim not in {2, 3}:
            raise ValueError("Input image must have 2 (grayscale) or 3 (multi-channel) dimensions.")

        # Calculate the target dimensions
        target_shape = (
            image.shape[0],
            image.shape[1] // self.downsample_factor,
            image.shape[2] // self.downsample_factor
        ) if image.ndim == 3 else (
            image.shape[0] // self.downsample_factor,
            image.shape[1] // self.downsample_factor
        )

        # Perform resizing with anti-aliasing to preserve quality
        downsampled_image = resize(
            image,
            output_shape=target_shape,
            anti_aliasing=True,
            preserve_range=True
        ).astype(image.dtype)

        return downsampled_image


# Example Usage
if __name__ == "__main__":
    # Example multi-channel image
    example_image = np.random.randint(0, 256, (3, 1024, 1024), dtype=np.uint8)

    # Initialize the downsampler
    downsampler = ImageDownsampler(downsample_factor=4)

    # Downsample the image
    downsampled_image = downsampler.downsample(example_image)

    # Print the result
    print(f"Original Shape: {example_image.shape}")
    print(f"Downsampled Shape: {downsampled_image.shape}")
