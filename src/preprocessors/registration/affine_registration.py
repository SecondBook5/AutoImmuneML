# File: src/preprocessors/registration/affine_registration.py
import numpy as np
import cv2
from typing import Tuple
from src.preprocessors.registration.base_registration import BaseRegistration
from src.preprocessors.feature_engineering.feature_extractor import FeatureExtractor
from src.preprocessors.feature_engineering.feature_matcher import FeatureMatcher
from src.preprocessors.utilities.image_warper import ImageWarper


class AffineRegistration(BaseRegistration):
    """
    Implements affine image registration using global transformations.

    **Concept**:
    - Affine registration aligns two images by estimating a 2x3 transformation matrix
      that supports scaling, rotation, translation, and shearing.
    """

    def __init__(self, feature_method: str = "ORB", matcher_method: str = "BF"):
        """
        Initialize the AffineRegistration class with feature and matcher methods.

        Args:
            feature_method (str): Method for feature detection and description (default: ORB).
            matcher_method (str): Method for feature matching (default: BF).
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(method=feature_method)
        self.feature_matcher = FeatureMatcher(method=matcher_method)

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using affine transformations.

        Args:
            image1 (np.ndarray): Reference image (aligned to the coordinate system).
            image2 (np.ndarray): Target image (to be aligned).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Affinely registered image.
                - 2x3 affine transformation matrix.
        """
        # Step 1: Preprocess images using the base class method
        image1_gray = self.preprocess_image(image1)
        image2_gray = self.preprocess_image(image2)

        # Step 2: Detect features and compute descriptors
        keypoints1, descriptors1 = self.feature_extractor.detect_and_describe(image1_gray)
        keypoints2, descriptors2 = self.feature_extractor.detect_and_describe(image2_gray)

        # Step 3: Match descriptors
        matches = self.feature_matcher.match_features(descriptors1, descriptors2)

        # Step 4: Extract matched points
        points1, points2 = self.extract_matched_points(keypoints1, keypoints2, matches)

        # Step 5: Estimate the affine transformation matrix using RANSAC
        affine_matrix, inliers = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)

        # Step 6: Apply the affine transformation using ImageWarper
        aligned_image = ImageWarper.warp_affine(image2, affine_matrix, output_shape=image1.shape)

        return aligned_image, affine_matrix

    @staticmethod
    def extract_matched_points(
        keypoints1: list, keypoints2: list, matches: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched keypoints from two images.

        Args:
            keypoints1 (list): Keypoints from the reference image.
            keypoints2 (list): Keypoints from the target image.
            matches (list): Matches between descriptors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Coordinates of matched keypoints in both images.
        """
        # Extract coordinates of matched keypoints in the reference and target images
        points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
        points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
        return points1, points2

    def evaluate_registration(self, image1: np.ndarray, image2: np.ndarray, aligned_image: np.ndarray) -> None:
        """
        Evaluate the quality of registration using error metrics from BaseRegistration.

        Args:
            image1 (np.ndarray): Reference image.
            image2 (np.ndarray): Target image before registration.
            aligned_image (np.ndarray): Target image after registration.
        """
        for metric in self.error_metrics.keys():
            error = self.compute_error(image1, aligned_image, metric)
            print(f"{metric}: {error:.4f}")
