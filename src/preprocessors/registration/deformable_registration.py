# File: src/preprocessors/registration/deformable_registration.py
import numpy as np
import cv2
from typing import Tuple
from skimage.registration import optical_flow_tvl1  # For computing displacement fields
from src.preprocessors.registration.base_registration import BaseRegistration
from src.preprocessors.utilities.image_warper import ImageWarper
from src.preprocessors.registration.affine_registration import AffineRegistration


class DeformableRegistration(BaseRegistration):
    """
    Performs deformable image registration for local, non-linear adjustments.

    **Concept**:
    - Combines global alignment (affine transformations) with local adjustments
      (displacement fields) to refine alignment.
    """

    def __init__(self, feature_method: str = "ORB", matcher_method: str = "BF"):
        """
        Initialize the DeformableRegistration class.

        Args:
            feature_method (str): Method for feature detection and description (default: ORB).
            matcher_method (str): Method for feature matching (default: BF).
        """
        super().__init__()
        self.affine_registration = AffineRegistration(feature_method, matcher_method)

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform deformable registration by combining global affine alignment with local adjustments.

        Args:
            image1 (np.ndarray): Reference image (aligned to the coordinate system).
            image2 (np.ndarray): Target image (to be aligned).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Deformably registered image.
                - Displacement field (pixel-wise shifts).
        """
        # Step 1: Perform global alignment using affine registration
        affine_aligned_image, affine_matrix = self.affine_registration.register_images(image1, image2)

        # Step 2: Compute the displacement field for local adjustments
        displacement_field = self.compute_displacement_field(image1, affine_aligned_image)

        # Step 3: Apply the displacement field to refine the alignment
        deformably_aligned_image = self.apply_displacement_field(affine_aligned_image, displacement_field)

        return deformably_aligned_image, displacement_field

    def compute_displacement_field(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute the displacement field between the reference and target images.

        **Concept**:
        - The displacement field encodes the local shifts required to align each pixel
          in the target image with the reference image.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image after global alignment.

        Returns:
            np.ndarray: Displacement field with shape (H, W, 2), where each pixel has [dx, dy].
        """
        # Preprocess both images to ensure consistency
        image1_gray = self.preprocess_image(image1)
        image2_gray = self.preprocess_image(image2)

        # Compute the optical flow (displacement field) using TV-L1 method
        displacement_field = optical_flow_tvl1(image1_gray, image2_gray)

        return displacement_field

    def apply_displacement_field(self, image: np.ndarray, displacement_field: np.ndarray) -> np.ndarray:
        """
        Warp the target image using the computed displacement field.

        **Concept**:
        - Applies pixel-wise displacements to the target image to refine alignment.

        Args:
            image (np.ndarray): Target image to deform.
            displacement_field (np.ndarray): Displacement field containing pixel-wise shifts.

        Returns:
            np.ndarray: Deformably registered image.
        """
        # Use ImageWarper to apply the displacement field to the image
        return ImageWarper.warp_with_displacement(image, displacement_field)
