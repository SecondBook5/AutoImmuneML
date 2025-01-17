# File: src/preprocessors/registration/image_loader.py
import os
import cv2
import numpy as np
from typing import List, Tuple, Union


class ImageLoader:
    """
    Handles loading and preprocessing of images for registration pipelines.

    Supports common image formats (e.g., .png, .tiff, .zarr) and applies optional preprocessing
    such as resizing and normalization.
    """

    def __init__(self, normalize: bool = True, resize_shape: Tuple[int, int] = None):
        """
        Initialize the ImageLoader with optional preprocessing settings.

        Args:
            normalize (bool): Whether to normalize pixel values to the range [0, 1].
            resize_shape (Tuple[int, int]): Desired shape for resizing images (height, width).
        """
        self.normalize = normalize
        self.resize_shape = resize_shape

    def load_image(self, file_path: str, to_gray: bool = False) -> np.ndarray:
        """
        Load a single image from a file and apply preprocessing.

        Args:
            file_path (str): Path to the image file.
            to_gray (bool): Whether to convert the image to grayscale.

        Returns:
            np.ndarray: Preprocessed image.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the image format is unsupported or loading fails.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Load the image using OpenCV
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")

        # Convert to grayscale if required
        if to_gray and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image if a resize shape is specified
        if self.resize_shape:
            image = cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to [0, 1] if required
        if self.normalize:
            image = image.astype('float32') / 255.0

        return image

    def load_images(self, file_paths: List[str], to_gray: bool = False) -> List[np.ndarray]:
        """
        Load multiple images from a list of file paths.

        Args:
            file_paths (List[str]): List of image file paths to load.
            to_gray (bool): Whether to convert each image to grayscale.

        Returns:
            List[np.ndarray]: List of preprocessed images.
        """
        images = []
        for file_path in file_paths:
            images.append(self.load_image(file_path, to_gray))
        return images
