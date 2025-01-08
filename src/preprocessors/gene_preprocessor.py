# File: src/preprocessors/gene_preprocessor.py
import scanpy as sc  # For handling and preprocessing gene expression data
import pandas as pd  # For tabular data representation
from sklearn.cluster import KMeans  # For clustering
from typing import Dict, Union  # For type annotations
from src.preprocessors.base_preprocessor import BasePreprocessor  # Base class for preprocessors


class GenePreprocessor(BasePreprocessor):
    """
    Preprocessor for gene expression data to prepare it for downstream analysis and modeling.

    **Concepts**:
    - Gene expression data contains high dimensionality and noise, requiring normalization
      and dimensionality reduction for better interpretability and computational efficiency.
    - Dimensionality reduction (PCA, t-SNE, UMAP) enhances exploratory analysis and clustering.
    - Clustering groups similar cells based on expression profiles to infer biological insights.

    Features:
    - Validation of input data integrity before processing.
    - Normalization of raw counts to account for sequencing depth differences.
    - Dimensionality reduction using PCA, t-SNE, and UMAP.
    - K-means clustering for identifying cell populations.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the GenePreprocessor.

        Args:
            output_dir (str): Directory to save processed data.
        """
        # Call the parent class constructor to initialize the output directory
        super().__init__(output_dir)

    def validate(self, sdata: sc.AnnData) -> bool:
        """
        Validate the input AnnData object for required attributes.

        Args:
            sdata (sc.AnnData): Input AnnData object containing gene expression data.

        Returns:
            bool: True if validation passes, False otherwise.

        Raises:
            ValueError: If the input data is missing required attributes or is invalid.
        """
        # Ensure the AnnData object contains the required attributes
        if not hasattr(sdata, "var") or not hasattr(sdata, "layers"):
            raise ValueError("Input data is missing required attributes `var` and/or `layers`.")
        if "counts" not in sdata.layers:
            raise ValueError("The input data does not contain the expected `counts` layer.")
        if "gene_symbols" not in sdata.var:
            raise ValueError("The input data does not contain `gene_symbols` in `var`.")

        # Return True if validation is successful
        return True

    def preprocess(self, sdata: sc.AnnData, **kwargs) -> Dict[str, Union[pd.DataFrame, sc.AnnData]]:
        """
        Preprocess gene expression data with normalization, dimensionality reduction, and clustering.

        Args:
            sdata (sc.AnnData): Input AnnData object containing gene expression data.
            kwargs: Additional arguments, such as `output_filename` for saving processed data.

        Returns:
            Dict[str, Union[pd.DataFrame, sc.AnnData]]: Processed outputs including:
                - Raw counts
                - PCA results
                - t-SNE results
                - UMAP results
                - Cluster labels

        Raises:
            ValueError: If input validation fails.
            RuntimeError: For any processing step failures.
        """
        # Step 1: Validate the input data integrity
        self.validate(sdata)

        # Step 2: Extract raw counts and gene names
        try:
            print("Extracting raw counts and gene names...")
            # Extract gene names from the input AnnData object
            gene_name_list = sdata.var['gene_symbols'].values

            # Extract raw counts as a DataFrame for downstream use
            raw_counts_df = pd.DataFrame(sdata.layers['counts'], columns=gene_name_list)
        except Exception as e:
            raise RuntimeError(f"Failed to extract raw counts or gene names: {e}")

        # Step 3: Normalize and filter the raw data
        try:
            print("Filtering and normalizing data...")
            normalized_adata = self._filter_and_normalize(sdata)
        except Exception as e:
            raise RuntimeError(f"Failed during filtering and normalization: {e}")

        # Step 4: Perform dimensionality reduction
        try:
            # Extract PCA-transformed data for dimensionality reduction.
            print("Applying PCA...")
            pca_results = self._apply_pca(normalized_adata)

            # Extract t-SNE results for non-linear dimensionality reduction.
            print("Applying t-SNE...")
            tsne_results = self._apply_tsne(normalized_adata)

            # Extract UMAP results for dimensionality reduction.
            print("Applying UMAP...")
            umap_results = self._apply_umap(normalized_adata)
        except Exception as e:
            raise RuntimeError(f"Failed during dimensionality reduction: {e}")

        # Step 5: Perform K-means clustering for exploratory cell population analysis.
        try:
            print("Performing clustering...")
            clusters = self._apply_clustering(normalized_adata)
        except Exception as e:
            raise RuntimeError(f"Failed during clustering: {e}")

        # Step 6: Save processed outputs
        output_filename = kwargs.get("output_filename", "preprocessed_genes.h5ad")
        try:
            self.save(normalized_adata, output_filename)
        except Exception as e:
            raise RuntimeError(f"Failed to save processed data: {e}")

        # Return processed results in a dictionary
        return {
            "raw_counts": raw_counts_df,
            "pca": pca_results,
            "tsne": tsne_results,
            "umap": umap_results,
            "clusters": clusters,
        }

    def _filter_and_normalize(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Normalize and filter gene expression data to prepare for downstream analysis.

        **Concepts**:
        - Filters out genes and cells with insufficient expression data to reduce noise.
        - Normalization adjusts expression values for sequencing depth differences.
        - Log transformation reduces skewness for better interpretability.

        Args:
            adata (sc.AnnData): Input AnnData object containing raw expression data.

        Returns:
            sc.AnnData: Filtered and normalized AnnData object.
        """
        # Remove genes expressed in fewer than 5 cells
        sc.pp.filter_genes(adata, min_counts=5)

        # Remove cells with fewer than 200 genes expressed
        sc.pp.filter_cells(adata, min_genes=200)

        # Normalize the total gene expression per cell to 10,000 counts
        sc.pp.normalize_total(adata, target_sum=1e4)

        # Apply log1p transformation to stabilize variance
        sc.pp.log1p(adata)

        # Return the filtered and normalized data
        return adata

    def _apply_pca(self, adata: sc.AnnData, n_components: int = 50) -> pd.DataFrame:
        """
        Perform PCA for dimensionality reduction.

        **Concepts**:
        - Reduces dimensionality by identifying directions of maximum variance.
        - Improves computational efficiency and reduces noise.

        Args:
            adata (sc.AnnData): Normalized AnnData object.
            n_components (int): Number of principal components to compute.

        Returns:
            pd.DataFrame: PCA-transformed data.
        """
        # Run PCA to compute the top `n_components` principal components
        sc.tl.pca(adata, n_comps=n_components)

        # Convert the PCA results into a DataFrame for easier handling
        return pd.DataFrame(
            adata.obsm["X_pca"],
            index=adata.obs.index,
            columns=[f"PC{i + 1}" for i in range(n_components)]
        )

    def _apply_tsne(self, adata: sc.AnnData, n_components: int = 2, perplexity: int = 30) -> pd.DataFrame:
        """
        Apply t-SNE for non-linear dimensionality reduction.

        **Concepts**:
        - Captures local structure of high-dimensional data for visualization.
        - Useful for identifying clusters in complex datasets.

        Args:
            adata (sc.AnnData): AnnData object after PCA.
            n_components (int): Number of dimensions for t-SNE.
            perplexity (int): Perplexity parameter for balancing local/global structure.

        Returns:
            pd.DataFrame: t-SNE-transformed data.
        """
        # Apply t-SNE on the PCA-transformed data
        sc.tl.tsne(adata, n_pcs=50, perplexity=perplexity)

        # Convert the t-SNE results into a DataFrame for easier handling
        return pd.DataFrame(
            adata.obsm["X_tsne"],
            index=adata.obs.index,
            columns=[f"tSNE_{i + 1}" for i in range(n_components)]
        )

    def _apply_umap(self, adata: sc.AnnData, n_components: int = 2) -> pd.DataFrame:
        """
        Apply UMAP for dimensionality reduction.

        **Concepts**:
        - Preserves both global and local structures in high-dimensional data.
        - Preferred over t-SNE for larger datasets due to speed and scalability.

        Args:
            adata (sc.AnnData): AnnData object after PCA.
            n_components (int): Number of dimensions for UMAP.

        Returns:
            pd.DataFrame: UMAP-transformed data.
        """
        # Apply UMAP on the PCA-transformed data
        sc.tl.umap(adata, n_components=n_components)

        # Convert the UMAP results into a DataFrame for easier handling
        return pd.DataFrame(
            adata.obsm["X_umap"],
            index=adata.obs.index,
            columns=[f"UMAP_{i + 1}" for i in range(n_components)]
        )

    def _apply_clustering(self, adata: sc.AnnData, n_clusters: int = 10) -> pd.Series:
        """
        Perform K-means clustering to group cells with similar expression profiles.

        **Concepts**:
        - Clusters identify groups of cells with shared biological characteristics.
        - Aids in exploratory analysis of cell subpopulations.

        Args:
            adata (sc.AnnData): AnnData object after PCA.
            n_clusters (int): Number of clusters.

        Returns:
            pd.Series: Cluster labels for each cell.
        """
        # Fit K-means clustering on the PCA-transformed data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(adata.obsm["X_pca"])

        # Store the cluster labels in the AnnData object's observations
        adata.obs["kmeans_clusters"] = cluster_labels

        # Return the cluster labels as a Pandas Series
        return pd.Series(cluster_labels, index=adata.obs.index)
