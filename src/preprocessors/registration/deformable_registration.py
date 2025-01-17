# File: src/preprocessors/registration/deformable_registration.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from src.preprocessors.registration.base_registration import BaseRegistration


class DeformableRegistration(BaseRegistration):
    """
    Performs deformable image registration for local, non-linear adjustments.
    Refines alignment by estimating and applying a displacement field or a thin-plate spline transformation.

    This method builds on top of affine registration to account for small-scale distortions
    and local mismatches between two images.
    """

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using deformable transformations.

        Args:
            image1 (np.ndarray): Reference image (aligned to the coordinate system).
            image2 (np.ndarray): Target image (to be aligned).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Deformably registered image.
                - Displacement field or transformation parameters applied.
        """
        # Preprocess both images to grayscale for feature extraction
        image1_gray = self.preprocess_image(image1)
        image2_gray = self.preprocess_image(image2)

        # Perform initial alignment using an affine transformation (optional, if not already done)
        affine_matrix = self.estimate_initial_affine(image1_gray, image2_gray)
        initial_aligned_image = self.warp_image(image2, affine_matrix, image1.shape)

        # Compute displacement field for fine-grained local adjustments
        displacement_field = self.compute_displacement_field(image1_gray, initial_aligned_image)

        # Apply the displacement field to warp the target image
        deformable_registered_image = self.apply_displacement_field(initial_aligned_image, displacement_field)

        return deformable_registered_image, displacement_field

    def estimate_initial_affine(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Estimate the initial affine transformation as a starting point for deformable registration.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image.

        Returns:
            np.ndarray: Affine transformation matrix.
        """
        # Use ORB features and RANSAC-based matching from AffineRegistration
        keypoints1, descriptors1 = self.detect_features(image1)
        keypoints2, descriptors2 = self.detect_features(image2)
        matches = self.match_features(descriptors1, descriptors2)
        points1, points2 = self.extract_matched_points(keypoints1, keypoints2, matches)

        # Estimate affine matrix using matched points
        affine_matrix = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)[0]
        return affine_matrix

    def compute_displacement_field(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute the displacement field between the reference and target images.

        The displacement field encodes the local shifts required to align each pixel in the target
        image to the reference image.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image after initial alignment.

        Returns:
            np.ndarray: Displacement field (shape: [H, W, 2]) where each pixel has [dx, dy].
        """
        # Use OpenCV's Dense Optical Flow (Farneback) for displacement estimation
        flow = cv2.calcOpticalFlowFarneback(
            image1, image2,
            None,  # Initial flow (None for fresh computation)
            0.5,   # Pyramid scale
            3,     # Number of pyramid levels
            15,    # Window size
            3,     # Iterations per level
            5,     # Polynomial expansion size
            1.2,   # Gaussian standard deviation
            0      # Flags
        )
        return flow

    def apply_displacement_field(self, image: np.ndarray, displacement_field: np.ndarray) -> np.ndarray:
        """
        Warp the target image using the computed displacement field.

        Args:
            image (np.ndarray): Target image to deform.
            displacement_field (np.ndarray): Displacement field containing pixel-wise shifts.

        Returns:
            np.ndarray: Deformably registered image.
        """
        # Create a grid of coordinates representing the image
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply the displacement field to the grid
        map_x = grid_x + displacement_field[..., 0]
        map_y = grid_y + displacement_field[..., 1]

        # Warp the image using the remapped coordinates
        return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    def visualize_displacement_field(self, displacement_field: np.ndarray, image: np.ndarray) -> None:
        """
        Visualize the displacement field as a quiver plot overlaying the reference or target image.

        Args:
            displacement_field (np.ndarray): Displacement field containing pixel-wise shifts.
            image (np.ndarray): Reference or target image to overlay the displacement field on.
        """
        # Create a grid of coordinates for the quiver plot
        h, w = displacement_field.shape[:2]
        step = max(1, h // 40)  # Reduce the number of arrows for readability
        grid_y, grid_x = np.mgrid[0:h:step, 0:w:step]

        # Sample the displacement field at the grid points
        sampled_field = displacement_field[grid_y, grid_x]

        # Create a quiver plot to visualize the displacement field
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.quiver(
            grid_x, grid_y,
            sampled_field[..., 0], sampled_field[..., 1],
            color='red', angles='xy', scale_units='xy', scale=1
        )
        plt.title("Displacement Field Visualization")
        plt.axis("off")
        plt.show()
