import cv2
import numpy as np
from typing import List, Tuple


class FeatureMatcher:
    """
    Matches feature descriptors between two images for registration.

    Supports brute-force matching and FLANN-based matching, with optional filtering using
    Lowe's ratio test.
    """

    def __init__(self, method: str = "BF", cross_check: bool = True):
        """
        Initialize the feature matcher with a specified method.

        Args:
            method (str): Matching method. Supported options are:
                          - "BF": Brute-Force Matcher (default).
                          - "FLANN": Fast Library for Approximate Nearest Neighbors.
            cross_check (bool): Whether to enable cross-checking in BFMatcher.
                                Ensures matches are mutual.
        """
        self.method = method.upper()
        self.cross_check = cross_check
        self.matcher = self._initialize_matcher()

    def _initialize_matcher(self):
        """
        Initialize the feature matcher based on the specified method.

        Returns:
            Feature matcher object (e.g., BFMatcher, FLANN-based matcher).
        """
        if self.method == "BF":
            # Use Brute-Force Matcher with Hamming distance (default for ORB)
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.cross_check)
        elif self.method == "FLANN":
            # Use FLANN-based matcher with default settings
            index_params = {"algorithm": 1, "trees": 5}  # KD-Tree for SIFT/SURF
            search_params = {"checks": 50}  # Number of checks for nearest neighbor search
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported method '{self.method}'. Supported methods: BF, FLANN.")

    def match_features(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match feature descriptors between two images.

        Args:
            descriptors1 (np.ndarray): Descriptors from the first image.
            descriptors2 (np.ndarray): Descriptors from the second image.

        Returns:
            List[cv2.DMatch]: List of matches sorted by distance.
        """
        # Match descriptors using the specified matcher
        matches = self.matcher.match(descriptors1, descriptors2)

        # Sort matches by distance (quality of match)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def knn_match_features(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray, k: int = 2, ratio: float = 0.75
    ) -> List[cv2.DMatch]:
        """
        Perform k-Nearest Neighbors (k-NN) matching with Lowe's ratio test.

        Args:
            descriptors1 (np.ndarray): Descriptors from the first image.
            descriptors2 (np.ndarray): Descriptors from the second image.
            k (int): Number of nearest neighbors to consider.
            ratio (float): Ratio for Lowe's ratio test to filter matches.

        Returns:
            List[cv2.DMatch]: Filtered list of matches passing the ratio test.
        """
        # Perform k-NN matching
        knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=k)

        # Apply Lowe's ratio test to filter matches
        good_matches = [
            m[0] for m in knn_matches if len(m) == k and m[0].distance < ratio * m[1].distance
        ]

        return good_matches

    def visualize_matches(
        self,
        image1: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        image2: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches: int = 50,
        output_path: str = None,
    ) -> None:
        """
        Visualize matched features between two images.

        Args:
            image1 (np.ndarray): First input image.
            keypoints1 (List[cv2.KeyPoint]): Keypoints from the first image.
            image2 (np.ndarray): Second input image.
            keypoints2 (List[cv2.KeyPoint]): Keypoints from the second image.
            matches (List[cv2.DMatch]): List of matches to visualize.
            max_matches (int): Maximum number of matches to display.
            output_path (str): Optional path to save the visualization.
        """
        # Draw the top matches
        matched_image = cv2.drawMatches(
            image1, keypoints1, image2, keypoints2, matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Display the matched image
        cv2.imshow("Matched Features", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the visualization if an output path is provided
        if output_path:
            cv2.imwrite(output_path, matched_image)
