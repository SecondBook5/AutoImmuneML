import cv2
import numpy as np
from typing import Tuple, List


class FeatureExtractor:
    """
    Detects keypoints and computes descriptors in images for feature-based registration.

    Supports traditional methods like ORB and SIFT as well as deep-learning-based methods
    like SuperPoint (if integrated).
    """

    def __init__(self, method: str = "ORB"):
        """
        Initialize the feature extractor with a specified method.

        Args:
            method (str): Feature detection method. Supported options are:
                          - "ORB": Oriented FAST and Rotated BRIEF (default).
                          - "SIFT": Scale-Invariant Feature Transform.
                          - "SuperPoint": Deep learning-based keypoint detector (requires additional setup).
        """
        self.method = method.upper()
        self.detector = self._initialize_detector()

    def _initialize_detector(self):
        """
        Initialize the feature detector based on the specified method.

        Returns:
            Feature detector object (e.g., ORB, SIFT).
        """
        if self.method == "ORB":
            return cv2.ORB_create()
        elif self.method == "SIFT":
            return cv2.SIFT_create()
        else:
            raise ValueError(f"Unsupported method '{self.method}'. Supported methods: ORB, SIFT.")

    def detect_and_describe(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors in the input image.

        Args:
            image (np.ndarray): Grayscale input image.

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]:
                - List of detected keypoints.
                - Array of feature descriptors corresponding to the keypoints.
        """
        # Validate the input image
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale.")

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        # Check if keypoints or descriptors were found
        if not keypoints:
            raise RuntimeError("No keypoints detected in the image.")

        return keypoints, descriptors

    def visualize_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], output_path: str = None) -> None:
        """
        Visualize keypoints detected in the image.

        Args:
            image (np.ndarray): Grayscale input image.
            keypoints (List[cv2.KeyPoint]): List of detected keypoints.
            output_path (str): Optional path to save the visualization.
        """
        # Draw keypoints on the image
        output_image = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Display the image with keypoints
        cv2.imshow("Keypoints", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the visualization if an output path is provided
        if output_path:
            cv2.imwrite(output_path, output_image)
