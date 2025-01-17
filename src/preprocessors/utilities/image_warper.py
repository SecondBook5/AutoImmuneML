# File: src/preprocessors/utilities/image_warper.py
import numpy as np
import cv2
from typing import Tuple


class ImageWarper:
    """
    Applies transformations to align images for registration.

    **Concept**:
    - Affine transformations: Linear transformations (scaling, rotation, translation, shearing).
    - Homography transformations: Non-linear perspective corrections.
    - Displacement fields: Pixel-wise transformations for local adjustments.
    """

    @staticmethod
    def warp_affine(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Apply an affine transformation to an image.

        Args:
            image (np.ndarray): Input image to warp.
            matrix (np.ndarray): 2x3 affine transformation matrix.
            output_shape (Tuple[int, int]): Shape of the output image (height, width).

        Returns:
            np.ndarray: Warped image.
        """
        # Use OpenCV's warpAffine function to apply the affine transformation
        # The `flags` argument specifies the interpolation method (default: linear)
        return cv2.warpAffine(image, matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)

    @staticmethod
    def warp_homography(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Apply a homography transformation to an image.

        Args:
            image (np.ndarray): Input image to warp.
            matrix (np.ndarray): 3x3 homography transformation matrix.
            output_shape (Tuple[int, int]): Shape of the output image (height, width).

        Returns:
            np.ndarray: Warped image.
        """
        # Use OpenCV's warpPerspective function to apply the homography transformation
        # Homography transformations allow for perspective corrections
        return cv2.warpPerspective(image, matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)

    @staticmethod
    def warp_with_displacement(
        image: np.ndarray, displacement_field: np.ndarray, interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Warp an image using a displacement field.

        **Concept**: Each pixel in the displacement field specifies how far to move the corresponding pixel
        in the input image.

        Args:
            image (np.ndarray): Input image to warp.
            displacement_field (np.ndarray): Displacement field with shape (H, W, 2), where each pixel
                                             contains [dx, dy] displacements.
            interpolation (int): Interpolation method to use (default: cv2.INTER_LINEAR).

        Returns:
            np.ndarray: Warped image.
        """
        # Extract the dimensions of the input image (height, width)
        h, w = image.shape[:2]

        # Create a grid of x and y coordinates for the image
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Add the displacement field to the grid coordinates
        # This shifts each pixel by the specified displacement values (dx, dy)
        map_x = (grid_x + displacement_field[..., 0]).astype(np.float32)
        map_y = (grid_y + displacement_field[..., 1]).astype(np.float32)

        # Use OpenCV's remap function to warp the image based on the remapped coordinates
        # `map_x` and `map_y` specify the new locations of each pixel in the output image
        return cv2.remap(image, map_x, map_y, interpolation)

    @staticmethod
    def warp_image(
        image: np.ndarray,
        transformation: np.ndarray,
        output_shape: Tuple[int, int],
        method: str = "affine"
    ) -> np.ndarray:
        """
        Generalized method to warp an image using the specified transformation method.

        Args:
            image (np.ndarray): Input image to warp.
            transformation (np.ndarray): Transformation matrix (2x3 for affine, 3x3 for homography).
            output_shape (Tuple[int, int]): Shape of the output image (height, width).
            method (str): Transformation method ('affine', 'homography').

        Returns:
            np.ndarray: Warped image.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        # Check if the transformation method is 'affine'
        if method == "affine":
            # Call warp_affine to apply the affine transformation
            return ImageWarper.warp_affine(image, transformation, output_shape)
        # Check if the transformation method is 'homography'
        elif method == "homography":
            # Call warp_homography to apply the homography transformation
            return ImageWarper.warp_homography(image, transformation, output_shape)
        else:
            # Raise an error for unsupported transformation methods
            raise ValueError(f"Unsupported transformation method: {method}")
