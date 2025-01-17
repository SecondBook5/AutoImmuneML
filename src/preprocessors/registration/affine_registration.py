# File: src/preprocessors/registration/affine_registration.py
import numpy as np
import cv2
from typing import Tuple
from src.preprocessors.registration.base_registration import BaseRegistration


class AffineRegistration(BaseRegistration):
    """
    Performs global affine registration, including scaling, rotation, and translation.
    Inherits shared functionality from BaseRegistration.

    This class aligns two images using affine transformations, which are linear transformations
    that preserve points, straight lines, and planes. Affine transformations include operations
    like scaling, rotation, translation, and shearing.

    The implementation uses:
    - ORB (Oriented FAST and Rotated BRIEF): A feature detection and description algorithm
      that identifies keypoints in images and computes descriptors for those keypoints. It is
      computationally efficient and robust to image scale and rotation.
    - BFMatcher (Brute-Force Matcher): A descriptor-matching algorithm that compares descriptors
      of keypoints between two images to find matches based on similarity.
    - RANSAC (Random Sample Consensus): An iterative method to estimate a model from a dataset
      that contains outliers. In this case, it is used to robustly estimate the affine
      transformation matrix by ignoring outlier matches.
    - cv2.estimateAffinePartial2D: A function from OpenCV to compute the affine transformation
      matrix between two sets of points.
    """

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using affine transformations.

        Args:
            image1 (np.ndarray): Reference image (the image to which the target will be aligned).
            image2 (np.ndarray): Target image (the image that will be transformed).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Registered target image after applying the affine transformation.
                - Affine transformation matrix used to align the target image to the reference image.
        """
        # Preprocess both images to grayscale for feature matching
        image1_gray = self.preprocess_image(image1)
        image2_gray = self.preprocess_image(image2)

        # Detect keypoints and compute descriptors using ORB
        keypoints1, descriptors1 = self.detect_features(image1_gray)
        keypoints2, descriptors2 = self.detect_features(image2_gray)

        # Match features between the two images
        matches = self.match_features(descriptors1, descriptors2)

        # Extract matched keypoints for transformation matrix estimation
        points1, points2 = self.extract_matched_points(keypoints1, keypoints2, matches)

        # Estimate the affine transformation matrix using RANSAC
        transformation_matrix = self.estimate_affine_matrix(points1, points2)

        # Warp the target image using the estimated transformation matrix
        registered_image = self.warp_image(image2, transformation_matrix, image1.shape)

        # Compute the alignment error for debugging purposes
        error = self.compute_error(image1_gray, registered_image, metric='MSE')
        print(f"Registration MSE: {error}")

        # Save the registered image
        self.save_image(registered_image, "registered_image.png")

        return registered_image, transformation_matrix

    def detect_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Detect keypoints and compute descriptors in an image using ORB.

        ORB (Oriented FAST and Rotated BRIEF) is a feature detection and description algorithm.
        It is efficient for identifying keypoints in an image (distinctive locations like corners)
        and computing descriptors that encode their visual appearance for matching.

        Args:
            image (np.ndarray): Input image to detect features from.

        Returns:
            Tuple[list, np.ndarray]:
                - Keypoints: A list of cv2.KeyPoint objects representing distinctive image features.
                - Descriptors: A numpy array of binary feature descriptors for the keypoints.
        """
        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> list:
        """
        Match features between two sets of descriptors using BFMatcher.

        BFMatcher (Brute-Force Matcher) compares descriptors between two images and finds the best
        matches. It calculates the Hamming distance (number of differing bits) between descriptors
        to determine similarity.

        Args:
            descriptors1 (np.ndarray): Descriptors from the reference image.
            descriptors2 (np.ndarray): Descriptors from the target image.

        Returns:
            list: A list of cv2.DMatch objects representing the best matches between descriptors.
        """
        # Initialize BFMatcher with Hamming distance metric
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance (lower distance indicates better matches)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def extract_matched_points(self, keypoints1: list, keypoints2: list, matches: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched points from keypoints using the matches.

        Args:
            keypoints1 (list): Keypoints from the reference image.
            keypoints2 (list): Keypoints from the target image.
            matches (list): Matched features between the two images.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Points1: Coordinates of matched keypoints in the reference image.
                - Points2: Coordinates of matched keypoints in the target image.
        """
        # Extract coordinates of matched keypoints in the reference and target images
        points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
        points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
        return points1, points2

    def estimate_affine_matrix(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        Estimate the affine transformation matrix using matched points.

        This method uses RANSAC (Random Sample Consensus) to robustly compute the affine
        transformation matrix. RANSAC iteratively selects subsets of matched points, estimates
        the transformation, and identifies inliers that agree with the estimated model.

        Args:
            points1 (np.ndarray): Matched points in the reference image.
            points2 (np.ndarray): Matched points in the target image.

        Returns:
            np.ndarray: Estimated affine transformation matrix.
        """
        # Estimate affine transformation matrix with RANSAC to ignore outliers
        transformation_matrix, _ = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)
        return transformation_matrix

    def warp_image(self, image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Apply an affine transformation to an image.

        Args:
            image (np.ndarray): Input image to warp.
            matrix (np.ndarray): Affine transformation matrix.
            output_shape (Tuple[int, int, int]): Shape of the reference image.

        Returns:
            np.ndarray: Warped (registered) image.
        """
        # Warp the image based on the affine transformation matrix
        return cv2.warpAffine(image, matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)
