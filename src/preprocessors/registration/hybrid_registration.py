# File: src/preprocessors/registration/hybrid_registration.py
from typing import Tuple
import numpy as np
from .affine_registration import AffineRegistration
from .deformable_registration import DeformableRegistration
from .base_registration import BaseRegistration


class HybridRegistration(BaseRegistration):
    """
    Combines affine and deformable registration for robust global and local image alignment.
    This module orchestrates the sequential application of global affine transformations
    and local deformable adjustments to achieve precise image registration.
    """

    def __init__(self):
        """
        Initialize the hybrid registration process by instantiating affine and deformable registration objects.
        """
        super().__init__()
        self.affine_registration = AffineRegistration()
        self.deformable_registration = DeformableRegistration()

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perform hybrid registration by first applying affine registration, followed by deformable registration.

        Args:
            image1 (np.ndarray): Reference image (the image to which the target will be aligned).
            image2 (np.ndarray): Target image (the image that will be transformed).

        Returns:
            Tuple[np.ndarray, dict]:
                - Fully registered image.
                - Registration details containing:
                    - 'affine_matrix': The affine transformation matrix.
                    - 'displacement_field': The computed displacement field for deformable registration.
        """
        # Step 1: Perform affine registration
        print("Starting affine registration...")
        affine_registered_image, affine_matrix = self.affine_registration.register_images(image1, image2)

        # Step 2: Perform deformable registration
        print("Starting deformable registration...")
        deformable_registered_image, displacement_field = self.deformable_registration.register_images(image1, affine_registered_image)

        # Step 3: Return the final registered image and details
        registration_details = {
            'affine_matrix': affine_matrix,
            'displacement_field': displacement_field
        }
        return deformable_registered_image, registration_details

    def compute_error(self, image1: np.ndarray, image2: np.ndarray, metric: str = 'MSE') -> float:
        """
        Compute the alignment error between two images using a specified metric.

        This method overrides the base class to provide better insight into hybrid registration quality.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image (aligned).
            metric (str): Error metric to use ('MSE' or 'SSIM').

        Returns:
            float: Computed error value.
        """
        # Call the base method to compute alignment error
        return super().compute_error(image1, image2, metric)

    def visualize_registration_steps(self, image1: np.ndarray, affine_image: np.ndarray, final_image: np.ndarray) -> None:
        """
        Visualize the intermediate steps of hybrid registration: original, affine, and final (deformable).

        Args:
            image1 (np.ndarray): Reference image.
            affine_image (np.ndarray): Image after affine registration.
            final_image (np.ndarray): Final registered image after deformable adjustments.
        """
        import matplotlib.pyplot as plt

        # Plot the images side by side for comparison
        plt.figure(figsize=(15, 5))
        titles = ['Reference Image', 'After Affine Registration', 'Final Registered Image']
        images = [image1, affine_image, final_image]

        for i, (title, img) in enumerate(zip(titles, images), 1):
            plt.subplot(1, 3, i)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        plt.suptitle("Hybrid Registration Steps")
        plt.show()
