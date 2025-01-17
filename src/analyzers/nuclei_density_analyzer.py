import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.colors import LogNorm
from skimage.filters import threshold_otsu


class NucleiDensityAnalyzer:
    """
    Analyzes nuclei density and generates density heatmaps, clusters, and features.

    This module supports feature extraction, density heatmap generation, and advanced clustering
    for biological images containing nuclei.
    """

    def __init__(self, nuclei_image: np.ndarray):
        """
        Initialize the analyzer with a binary nuclei image.

        Args:
            nuclei_image (np.ndarray): Binary image where non-zero pixels represent nuclei regions.
        """
        # Store the binary nuclei image provided by the user
        self.nuclei_image = nuclei_image

        # Label connected components in the binary image to identify individual nuclei
        self.labeled_nuclei = label(nuclei_image)

        # Extract centroids (geometric centers) of all labeled nuclei regions
        self.centroids = np.array([region.centroid for region in regionprops(self.labeled_nuclei)])

        # Raise an error if no centroids are detected, ensuring the input is valid
        if self.centroids.size == 0:
            raise ValueError("No nuclei detected in the image.")

    def generate_density_heatmap(
        self,
        bins=300,
        bandwidth=30,
        colormap="viridis",
        threshold_method="percentile",
        threshold_percentile=90,
        apply_smoothing=True
    ) -> np.ndarray:
        """
        Generate a density heatmap using Kernel Density Estimation (KDE) and create a high-density mask.

        Args:
            bins (int): Number of bins for the KDE grid.
            bandwidth (float): Bandwidth for KDE; controls smoothness.
            colormap (str): Colormap for heatmap visualization.
            threshold_method (str): Method for determining the threshold ("percentile" or "otsu").
            threshold_percentile (int): Percentile for determining the threshold (used if "percentile").
            apply_smoothing (bool): Whether to smooth the binary mask using morphological closing.

        Returns:
            np.ndarray: Binary mask of high-density regions.
        """
        # Get the dimensions of the nuclei image
        x_dim, y_dim = self.nuclei_image.shape[1], self.nuclei_image.shape[0]

        # Create grid points (X, Y) for KDE evaluation
        x = np.linspace(0, x_dim, bins)
        y = np.linspace(0, y_dim, bins)
        X, Y = np.meshgrid(x, y)

        # Initialize and train the Kernel Density Estimation model
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(self.centroids)

        # Flatten grid points and compute KDE scores for each point
        grid = np.vstack([Y.ravel(), X.ravel()]).T
        Z = np.exp(kde.score_samples(grid)).reshape(X.shape)

        # Normalize the density values to the range [0, 1] for consistent visualization
        Z /= Z.max()

        # Determine the threshold for high-density regions
        if threshold_method == "percentile":
            # Use the specified percentile to determine the threshold
            threshold = np.percentile(Z, threshold_percentile)
        elif threshold_method == "otsu":
            # Use Otsu's method to determine the threshold
            threshold = threshold_otsu(Z)
        else:
            raise ValueError(f"Unsupported thresholding method: {threshold_method}")

        # Create a binary mask for high-density regions
        high_density_mask = (Z > threshold).astype(np.uint8)

        # Optionally apply morphological smoothing to the binary mask
        if apply_smoothing:
            high_density_mask = binary_closing(high_density_mask, disk(3))

        # Plot the density heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(Z, extent=(0, x_dim, y_dim, 0), cmap=colormap, norm=LogNorm())
        plt.colorbar(label="Nuclei Density")
        plt.title("Density Heatmap")
        plt.show()

        # Plot the high-density mask
        plt.figure(figsize=(10, 10))
        plt.imshow(high_density_mask, cmap="gray", extent=(0, x_dim, y_dim, 0))
        plt.title("High-Density Mask")
        plt.show()

        # Return the binary mask
        return high_density_mask

    def extract_features(self) -> np.ndarray:
        """
        Extract nuclei features such as area, eccentricity, and compactness.

        Returns:
            np.ndarray: Array of extracted features (rows=nuclei, columns=features).
        """
        # Initialize an empty list to store features for each nucleus
        features = []

        # Loop through each labeled nucleus region to compute its features
        for region in regionprops(self.labeled_nuclei):
            # Compute compactness: area divided by the square of the perimeter
            compactness = (region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0

            # Append centroid, area, eccentricity, and compactness for each nucleus
            features.append([region.centroid[0], region.centroid[1], region.area, region.eccentricity, compactness])

        # Convert the features list to a NumPy array for further analysis
        return np.array(features)

    def perform_clustering(self, features: np.ndarray, eps: float = 30, min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering on extracted features.

        Args:
            features (np.ndarray): Features array (rows=nuclei, columns=features).
            eps (float): Maximum distance between points to consider them as neighbors.
            min_samples (int): Minimum points required to form a dense cluster.

        Returns:
            np.ndarray: Cluster labels for each nucleus.
        """
        # Standardize features to ensure all features contribute equally to the clustering
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Initialize the DBSCAN clustering algorithm with specified parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # Fit the DBSCAN model to the features and obtain cluster labels
        cluster_labels = dbscan.fit_predict(features)

        # Return the cluster labels for each nucleus
        return cluster_labels

    def evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """
        Evaluate clustering performance using various metrics.

        Args:
            features (np.ndarray): Features array (rows=nuclei, columns=features).
            labels (np.ndarray): Cluster labels for each nucleus.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Handle cases with no or a single cluster (avoid calculating metrics)
        if len(set(labels)) <= 1:
            return {"Silhouette Score": -1, "Davies-Bouldin Index": -1, "Calinski-Harabasz Index": -1}

        # Calculate silhouette score: higher is better
        silhouette = silhouette_score(features, labels)

        # Calculate Davies-Bouldin index: lower is better
        db_index = davies_bouldin_score(features, labels)

        # Calculate Calinski-Harabasz score: higher is better
        ch_score = calinski_harabasz_score(features, labels)

        # Return clustering evaluation metrics as a dictionary
        return {
            "Silhouette Score": silhouette,
            "Davies-Bouldin Index": db_index,
            "Calinski-Harabasz Index": ch_score
        }
