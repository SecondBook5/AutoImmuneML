# File: src/preprocessors/gene_preprocessor.py
# File: src/preprocessors/gene_preprocessor.py

import os
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict
from src.preprocessors.base_preprocessor import BasePreprocessor


class GenePreprocessor(BasePreprocessor):
    """
    Preprocessor for gene expression data to prepare data for downstream analysis and modeling.

    **Concepts**:
    - Gene expression data contains high dimensionality and noise, requiring normalization
      and dimensionality reduction for better interpretability and computational efficiency.
    - Dimensionality reduction (PCA, t-SNE, UMAP) enhances exploratory analysis and clustering.
    - Clustering groups similar cells based on expression profiles to infer biological insights.

    Features:
    - Normalization of raw counts to account for sequencing depth differences.
    - Dimensionality reduction using PCA, t-SNE, and UMAP.
    - K-means clustering for identifying cell populations.
    """

    def preprocess(self, sdata, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Preprocess gene expression data with normalization, dimensionality reduction, and clustering.

        Args:
            sdata: A SpatialData object containing gene expression data (`anucleus`).
            kwargs: Additional arguments, such as `output_filename` for saving processed data.

        Returns:
            Dict[str, pd.DataFrame]: Processed outputs including:
                - Normalized AnnData
                - PCA results
                - t-SNE results
                - UMAP results
                - Cluster labels
        """
        # Extract gene names from `anucleus` for consistency in downstream analyses.
        print("Extracting gene names...")
        gene_name_list = sdata['anucleus'].var['gene_symbols'].values

        # Create DataFrame for raw counts from `anucleus` layers.
        print("Extracting raw counts...")
        raw_counts_df = pd.DataFrame(
            sdata['anucleus'].layers['counts'],
            columns=gene_name_list
        )

        # Normalize and filter the raw expression data.
        print("Filtering and normalizing data...")
        normalized_adata = self._filter_and_normalize(sdata['anucleus'])

        # Extract PCA-transformed data for dimensionality reduction.
        print("Applying PCA...")
        pca_results = self._apply_pca(normalized_adata)

        # Extract t-SNE results for non-linear dimensionality reduction.
        print("Applying t-SNE...")
        tsne_results = self._apply_tsne(normalized_adata)

        # Extract UMAP results for dimensionality reduction.
        print("Applying UMAP...")
        umap_results = self._apply_umap(normalized_adata)

        # Perform K-means clustering for exploratory cell population analysis.
        print("Performing clustering...")
        clusters = self._apply_clustering(normalized_adata)

        # Save the processed AnnData file for reuse.
        output_filename = kwargs.get("output_filename", "preprocessed_genes.h5ad")
        self.save(normalized_adata, output_filename)

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
        # Filter genes expressed in fewer than 5 cells.
        sc.pp.filter_genes(adata, min_counts=5)

        # Filter cells with fewer than 200 genes expressed.
        sc.pp.filter_cells(adata, min_genes=200)

        # Normalize expression counts to 10,000 counts per cell.
        sc.pp.normalize_total(adata, target_sum=1e4)

        # Apply log1p transformation to stabilize variance.
        sc.pp.log1p(adata)

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
            pd.DataFrame: DataFrame containing the top principal components.
        """
        # Perform PCA using AnnData's built-in function.
        sc.tl.pca(adata, n_comps=n_components)

        # Convert PCA results to a DataFrame for easy access.
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
            perplexity (int): Parameter controlling local-global tradeoff.

        Returns:
            pd.DataFrame: DataFrame containing the t-SNE-transformed data.
        """
        # Perform t-SNE using AnnData's built-in function.
        sc.tl.tsne(adata, n_pcs=50, perplexity=perplexity)

        # Convert t-SNE results to a DataFrame.
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
            pd.DataFrame: DataFrame containing the UMAP-transformed data.
        """
        # Perform UMAP using AnnData's built-in function.
        sc.tl.umap(adata, n_components=n_components)

        # Convert UMAP results to a DataFrame.
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
        # Fit K-means clustering on PCA-transformed data.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(adata.obsm["X_pca"])

        # Store cluster labels in AnnData's observations.
        adata.obs["kmeans_clusters"] = cluster_labels

        return pd.Series(cluster_labels, index=adata.obs.index)
