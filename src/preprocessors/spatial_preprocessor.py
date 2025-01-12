# File: src/preprocessors/spatial_preprocessor.py
import os  # For file and directory operations
import numpy as np  # For numerical operations
import pandas as pd  # For handling tabular data
from scipy.spatial import distance_matrix  # For computing pairwise distances
from typing import Dict, Any  # For type annotations
from src.preprocessors.base_preprocessor import BasePreprocessor  # Base class for preprocessors


class SpatialPreprocessor(BasePreprocessor):
    """
    Preprocessor for spatial data to compute spatial features for downstream analysis.

    Features:
    - Validation of spatial data integrity.
    - Computation of pairwise distances between nuclei.
    - Generation of adjacency matrices.
    - Saving features in structured formats for downstream tasks.
    """

    def preprocess(self, dataset: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main preprocessing pipeline for spatial data.

        Args:
            dataset (Dict[str, Any]): Dataset containing spatial information.
            kwargs: Additional arguments (e.g., distance threshold for adjacency).

        Returns:
            Dict[str, Any]: Computed spatial features (e.g., distances, adjacency).
        """
        # Validate spatial data integrity
        self.validate(dataset)

        # Extract centroids of nuclei from the dataset
        centroids = self._extract_centroids(dataset)

        # Extract cell IDs associated with each nucleus
        cell_ids = self._extract_cell_ids(dataset)

        # Compute pairwise distances between centroids
        distances = self._compute_distances(centroids)

        # Check if a distance threshold is provided for adjacency matrix computation
        threshold = kwargs.get("distance_threshold", None)
        adjacency_matrix = None  # Initialize adjacency matrix as None
        if threshold:
            # Compute adjacency matrix based on the threshold
            adjacency_matrix = self._compute_adjacency(distances, threshold)

        # Create an output dictionary containing computed spatial features
        output = {
            "centroids": centroids,
            "distances": distances,
            "adjacency_matrix": adjacency_matrix,
            "cell_ids": cell_ids
        }

        return output  # Return the computed spatial features

    def validate(self, dataset: Dict[str, Any]):
        """
        Validate the integrity of the spatial data.

        Args:
            dataset (Dict[str, Any]): Dataset containing spatial information.

        Raises:
            ValueError: If required attributes are missing or invalid.
        """
        # Check if the dataset contains the required key 'HE_nuc_registered'
        if "HE_nuc_registered" not in dataset:
            raise ValueError("Dataset is missing 'HE_nuc_registered' key.")

        # Ensure that 'HE_nuc_registered' is a NumPy array
        if not isinstance(dataset["HE_nuc_registered"], np.ndarray):
            raise ValueError("'HE_nuc_registered' must be a NumPy array.")

    def _extract_centroids(self, dataset: Dict[str, Any]) -> np.ndarray:
        """
        Extract centroids of nuclei from the dataset.

        Args:
            dataset (Dict[str, Any]): Dataset containing spatial information.

        Returns:
            np.ndarray: Array of nucleus centroids (x, y coordinates).
        """
        # Get the nucleus segmentation mask from the dataset
        nuc_mask = dataset["HE_nuc_registered"]

        # Import regionprops from skimage.measure for extracting region properties
        from skimage.measure import regionprops
        regions = regionprops(nuc_mask)  # Get properties of connected regions (nuclei)

        # Extract centroids from each region and return as a NumPy array
        centroids = np.array([region.centroid for region in regions])
        return centroids

    def _extract_cell_ids(self, dataset: Dict[str, Any]) -> np.ndarray:
        """
        Extract cell IDs from the nucleus segmentation data.

        Args:
            dataset (Dict[str, Any]): Dataset containing spatial information.

        Returns:
            np.ndarray: Array of cell IDs.
        """
        # Get the nucleus segmentation mask from the dataset
        nuc_mask = dataset["HE_nuc_registered"]

        # Import regionprops from skimage.measure for extracting region properties
        from skimage.measure import regionprops
        regions = regionprops(nuc_mask)  # Get properties of connected regions (nuclei)

        # Extract cell IDs (unique labels) for each nucleus
        cell_ids = np.array([region.label for region in regions])
        return cell_ids

    def _compute_distances(self, centroids: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between centroids.

        Args:
            centroids (np.ndarray): Array of centroid coordinates.

        Returns:
            np.ndarray: Pairwise distance matrix.
        """
        # Use scipy's distance_matrix to compute distances between all centroids
        return distance_matrix(centroids, centroids)

    def _compute_adjacency(self, distances: np.ndarray, threshold: float) -> np.ndarray:
        """
        Generate an adjacency matrix based on distance threshold.

        Args:
            distances (np.ndarray): Pairwise distance matrix.
            threshold (float): Distance threshold for adjacency.

        Returns:
            np.ndarray: Binary adjacency matrix.
        """
        # Create a binary matrix where entries are 1 if distance <= threshold, else 0
        adjacency_matrix = (distances <= threshold).astype(int)

        # Set the diagonal to 0 to avoid self-loops in the adjacency matrix
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix
