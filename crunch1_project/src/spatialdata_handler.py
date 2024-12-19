# File: crunch1_project/src/spatialdata_handler.py

import spatialdata as sd  # For handling spatial omics datasets
import matplotlib.pyplot as plt  # For static 2D visualizations
import plotly.express as px  # For interactive visualizations
import os  # For file and path operations
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and storage
from tqdm import tqdm  # For progress tracking
from concurrent.futures import ThreadPoolExecutor  # For multithreading
from skimage.transform import rescale  # For resampling large images
from typing import Optional, List, Dict  # For type annotations
import concurrent.futures  # For managing exceptions during multithreading
from threading import Lock # For thread-safe updates

class SpatialDataHandler:
    """
    A utility class for managing, validating, visualizing, and processing
    SpatialData objects loaded from Zarr files. This class provides features
    such as multithreaded dataset loading, validation of required components,
    data subsampling, image visualization, and extraction of nuclei and gene
    expression data.

    The class is designed to handle multiple datasets concurrently, making it
    scalable for projects involving large and complex SpatialData objects.

    Key Features:
    - Load datasets from multiple Zarr files in parallel using multithreading.
    - Validate datasets for required images and tables.
    - Subsample datasets to limit memory usage.
    - Extract nuclei coordinates and link them to gene expression profiles.
    - Support for lazy loading of images to optimize memory usage.
    - Generate static and interactive visualizations for data exploration.
    - Provide detailed summaries of loaded datasets, including statistics.

    Args:
        zarr_paths (List[str]): A list of file paths to the Zarr datasets.

    Attributes:
        zarr_paths (List[str]): The list of Zarr dataset paths provided during initialization.
        datasets (Dict[str, SpatialData]): A dictionary mapping dataset names to loaded SpatialData objects.
        lazy_loaded_images (Dict[str, np.ndarray]): A cache for lazily loaded images, indexed by dataset name and image key.
        lazy_loaded_tables (Dict[str, Any]): A cache for lazily loaded tables, indexed by dataset name and table key.
    """

    def __init__(self, zarr_paths: List[str]) -> None:
        """
        Initialize the SpatialDataHandler with a list of Zarr dataset paths.

        Args:
            zarr_paths (List[str]): A list of file paths to the Zarr datasets.
        """
        # Initialize the SpatialDataHandler with a list of Zarr dataset paths
        self.zarr_paths: List[str] = zarr_paths  # Store dataset paths
        self.datasets: Dict[str, sd.SpatialData] = {}  # Dictionary to store loaded SpatialData objects
        self.lazy_loaded_images: Dict[str, np.ndarray] = {}  # Cache for lazily loaded images
        self.lazy_loaded_tables: Dict[str, pd.DataFrame] = {}  # Cache for lazily loaded tables
        print(f"Initialized handler for datasets: {zarr_paths}")  # Log initialization

    def load_data(self, max_retries: int = 3) -> None:
        """
        Load all SpatialData objects from the provided Zarr files using multithreading.

        Args:
            max_retries (int): Maximum number of retries for failed dataset loads.

        Raises:
            FileNotFoundError: If a specified Zarr file is not found.
            Exception: If any dataset consistently fails to load after retries.
        """
        def load_single_dataset(zarr_path: str) -> None:
            """
            Load a single SpatialData object from a Zarr file with retry logic.

            Args:
                zarr_path (str): Path to the Zarr dataset.

            Raises:
                FileNotFoundError: If the file is not found.
                Exception: If the dataset fails to load after retries.
            """
            nonlocal lock  # Access the shared lock for thread-safe updates
            retries = 0
            while retries < max_retries:
                try:
                    if not os.path.exists(zarr_path):  # Check if the file exists
                        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

                    dataset_name = os.path.basename(zarr_path)  # Extract dataset name from path
                    print(f"Loading dataset: {dataset_name} (Attempt {retries + 1}/{max_retries})...")
                    dataset = sd.read_zarr(zarr_path)  # Load the SpatialData object

                    with lock:  # Ensure thread-safe updates to `self.datasets`
                        self.datasets[dataset_name] = dataset
                    print(f"Dataset '{dataset_name}' loaded successfully.")
                    return  # Exit the loop if loading succeeds

                except Exception as e:
                    retries += 1
                    print(f"Failed to load dataset '{zarr_path}' on attempt {retries}/{max_retries}: {e}")

            # If all retries fail, log the final error
            raise Exception(f"Dataset '{zarr_path}' failed to load after {max_retries} retries.")

        lock = Lock()  # Initialize a thread lock for safe access to shared resources
        errors = []  # List to track errors during dataset loading

        with ThreadPoolExecutor() as executor:
            # Submit tasks to load datasets concurrently
            futures = {executor.submit(load_single_dataset, path): path for path in self.zarr_paths}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.zarr_paths), desc="Loading Datasets"):
                try:
                    future.result()  # Wait for each task to complete
                except Exception as e:
                    errors.append((futures[future], e))  # Record errors for failed tasks

        # Log a summary of errors if any datasets failed to load
        if errors:
            print("\nSummary of Errors:")
            for path, error in errors:
                print(f" - {path}: {error}")

    def validate_data(self, required_images: Optional[List[str]] = None, required_tables: Optional[List[str]] = None,
                      strict: bool = True) -> None:
        """
        Validate the presence of required images and tables in all loaded datasets.

        Args:
            required_images (Optional[List[str]]): A list of required image keys.
            required_tables (Optional[List[str]]): A list of required table keys.
            strict (bool): Whether to raise a ValueError for missing components (default: True).

        Raises:
            ValueError: If a required component (image/table) is missing in any dataset and `strict` is True.
        """
        # Use default keys if none are provided
        required_images = required_images or ["HE_original", "HE_nuc_original"]
        required_tables = required_tables or ["anucleus"]

        # Iterate over all loaded datasets
        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Validating Datasets"):
            # Iterate over each loaded dataset
            print(f"Validating dataset: {dataset_name}...")  # Log validation start
            missing_components: List[str] = []  # Collect missing components

            # Check for missing images
            for key in required_images:
                if key not in dataset.images.keys():
                    missing_components.append(f"Image '{key}'")

            # Check for missing tables
            for key in required_tables:
                if key not in dataset.tables.keys():
                    missing_components.append(f"Table '{key}'")

            # Handle missing components
            if missing_components:
                error_message = (
                        f"Dataset '{dataset_name}' is missing the following components:\n" +
                        "\n".join(f" - {component}" for component in missing_components)
                )
                print(error_message)  # Log missing components

                if strict:
                    raise ValueError(error_message)  # Raise an error if strict mode is enabled
            else:
                print(f"Dataset '{dataset_name}' is fully validated.")  # Log success

    def subsample_data(self, max_cells: int = 1000) -> None:
        """
        Subsample all datasets to limit the number of cells for memory efficiency.

        Args:
            max_cells (int): Maximum number of cells to retain in each dataset.
        """
        # Subsample all loaded datasets to limit the number of cells
        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Subsampling Datasets"):
            # Access the anucleus table
            anucleus = dataset.tables.get("anucleus")
            if anucleus is None:  # Skip if the table is missing
                print(f"Dataset '{dataset_name}' does not contain an 'anucleus' table. Skipping subsampling.")
                continue

            # Ensure `shape` is valid and accessible
            if not hasattr(anucleus, "shape") or not isinstance(anucleus.shape, tuple):
                raise TypeError(f"Table 'anucleus' in dataset '{dataset_name}' does not have a valid shape attribute.")

            total_cells = anucleus.shape[0]  # Get the total number of cells
            if total_cells > max_cells:  # Check if subsampling is required
                print(f"Subsampling {total_cells} cells to {max_cells} in dataset '{dataset_name}'...")
                subsample_indices = np.random.choice(anucleus.obs_names, max_cells, replace=False)
                # Update the table with the subsampled data
                self.datasets[dataset_name].tables["anucleus"] = anucleus[subsample_indices, :]
                print(f"Subsampled dataset '{dataset_name}' to {max_cells} cells.")
            else:
                print(f"No subsampling needed for dataset '{dataset_name}'.")

    def extract_nuclei_and_gene_expression(self, gene_subset: Optional[List[str]] = None, nuclei_subset: Optional[List[str]] = None, batch_size: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Extract nuclei coordinates and link them to their gene expression data.

        Args:
            gene_subset (Optional[List[str]]): A subset of genes to extract.
            nuclei_subset (Optional[List[str]]): A subset of nuclei to include.
            batch_size (int): The number of genes to process in each batch for efficiency.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping dataset names to their corresponding dataframes.

            Raises:
                ValueError: If required components are missing or misaligned.
        """
        # Initialize results dictionary to store processed data for each dataset
        results: Dict[str, pd.DataFrame] = {}

        # Iterate over all datasets loaded into the handler
        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Extracting Data"):
            # Retrieve the anucleus table for nuclei and gene data
            anucleus = dataset.tables.get("anucleus")
            if anucleus is None:  # Skip dataset if anucleus table is missing
                print(f"Dataset '{dataset_name}' does not contain an 'anucleus' table. Skipping extraction.")
                continue

            # Extract spatial coordinates of nuclei
            nuclei_coords = anucleus.obsm.get("spatial")
            # Extract nuclei IDs and gene expression matrix
            nuclei_ids = anucleus.obs_names
            gene_expression = anucleus.X
            # Extract gene names associated with the expression matrix
            gene_names = anucleus.var_names

            # Perform integrity checks to ensure data alignment and presence
            if nuclei_coords is None or gene_expression is None:
                raise ValueError(f"Dataset '{dataset_name}' is missing required spatial or expression data.")
            if len(nuclei_ids) != gene_expression.shape[0]:
                raise ValueError(f"Dataset '{dataset_name}' has mismatched nuclei IDs and gene expression rows.")
            if len(gene_names) != gene_expression.shape[1]:
                raise ValueError(f"Dataset '{dataset_name}' has mismatched gene names and expression columns.")

            # Apply nuclei subset filter if provided
            if nuclei_subset:
                nuclei_mask = np.isin(nuclei_ids, nuclei_subset)
                nuclei_coords = nuclei_coords[nuclei_mask]
                nuclei_ids = nuclei_ids[nuclei_mask]

            # Apply gene subset filter if provided
            if gene_subset:
                gene_mask = np.isin(gene_names, gene_subset)
                gene_expression = gene_expression[:, gene_mask]
                gene_names = gene_names[gene_mask]

            # Calculate the number of batches to process genes efficiently
            num_genes = gene_expression.shape[1]
            num_batches = (num_genes + batch_size - 1) // batch_size

            # Initialize data dictionary to build the result DataFrame
            data: Dict[str, List] = {
                "nuclei_id": nuclei_ids,
                "x_coord": nuclei_coords[:, 0],
                "y_coord": nuclei_coords[:, 1],
            }

            # Process genes in batches to handle large datasets efficiently
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size  # Start index for the current batch
                end = min(start + batch_size, num_genes)  # End index for the current batch
                batch_gene_names = gene_names[start:end]  # Get names of genes in the current batch
                batch_expression = gene_expression[:, start:end]  # Get expression values for the batch

                # Add each gene's expression data to the dictionary
                for i, gene in tqdm(
                        enumerate(batch_gene_names),
                        desc=f"Processing Batch {batch_idx + 1}/{num_batches} in {dataset_name}",
                        total=len(batch_gene_names),
                ):
                    data[gene] = batch_expression[:, i]

            # Create a DataFrame from the processed data and add it to the results dictionary
            results[dataset_name] = pd.DataFrame(data)
            # Log completion for the dataset
            print(f"Extraction complete for dataset '{dataset_name}'.")

        # Return the processed results for all datasets
        return results

    def get_image(self, dataset_name: str, key: str) -> np.ndarray:
        """
       Retrieve an image from the specified dataset, using lazy loading.

       Args:
           dataset_name (str): The name of the dataset.
           key (str): The key of the image to retrieve.

       Returns:
           np.ndarray: The requested image as a NumPy array.

       Raises:
           ValueError: If the dataset is not found.
           KeyError: If the image key is not present in the dataset.
       """
        # Lazily load and retrieve an image from the cache or dataset
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")  # Raise error if dataset is missing
        cache_key = f"{dataset_name}:{key}"  # Generate a cache key
        if cache_key not in self.lazy_loaded_images:
            if key not in self.datasets[dataset_name].images.keys():
                raise KeyError(f"Image '{key}' not found in dataset '{dataset_name}'.")
            self.lazy_loaded_images[cache_key] = self.datasets[dataset_name].images[key].to_numpy()  # Cache the image
        return self.lazy_loaded_images[cache_key]  # Return the cached image

    def interactive_visualize(self, dataset_name: str, key: str) -> None:
        """
        Create an interactive visualization of a specified image.

        Args:
            dataset_name (str): The name of the dataset.
            key (str): The key of the image to visualize.

        Raises:
            ValueError: If the dataset is not found.
            KeyError: If the image key is not present in the dataset.
        """
        # Create an interactive visualization of an image using Plotly
        image = self.get_image(dataset_name, key)  # Use the lazy-loaded image
        fig = px.imshow(image, title=f"Interactive Visualization: {key} ({dataset_name})", color_continuous_scale="Viridis")
        fig.update_layout(coloraxis_colorbar={"title": "Intensity"})  # Configure the color bar
        fig.show()  # Display the plot

    @staticmethod
    def plot_image(image: np.ndarray, title: str, cmap: Optional[str] = None) -> None:
        """
        Display a 2D image using Matplotlib.

        Args:
            image (np.ndarray): The image data to display.
            title (str): The title for the plot.
            cmap (Optional[str]): Colormap for visualization (e.g., 'gray').

        Raises:
            ValueError: If the image data is invalid.
        """
        # Display a 2D image using Matplotlib
        if image is None:
            raise ValueError("Cannot plot: Image data is None.")  # Raise error for invalid data
        plt.figure(figsize=(8, 8))  # Create a new figure
        plt.imshow(image, cmap=cmap)  # Show the image with the specified colormap
        plt.title(title)  # Set the plot title
        plt.axis("off")  # Hide axes for better visualization
        plt.show()  # Display the plot

    @staticmethod
    def rescale_image(image: np.ndarray, rescale_factor: float) -> np.ndarray:
        """
        Rescale an image for memory efficiency or improved visualization.

        Args:
            image (np.ndarray): The image to rescale.
            rescale_factor (float): The factor by which to rescale the image.

        Returns:
            np.ndarray: The rescaled image.
        """
        # Rescale an image for visualization or memory efficiency
        return rescale(image, rescale_factor, anti_aliasing=True, multichannel=True)  # Perform rescaling

    def print_summary(self) -> None:
        """
       Print a detailed summary of all loaded SpatialData objects, including statistics
       and information about images and tables.

       Raises:
           ValueError: If no datasets are loaded.
       """
        # Print a detailed summary of the loaded SpatialData objects
        if not self.datasets:  # Raise error if no datasets are loaded
            raise ValueError("No datasets loaded. Please call `load_data` first.")
        for dataset_name, dataset in self.datasets.items():  # Iterate over each dataset
            print(f"Dataset: {dataset_name}")  # Print the dataset name
            print("- Images:")  # Print details about images
            for key, img in dataset.images.items():
                print(f"  {key}: shape {img.shape}, dtype {img.dtype}")
            print("- Tables:")  # Print details about tables
            for key, table in dataset.tables.items():
                print(f"  {key}: {table.shape[0]} rows, {table.shape[1]} columns")
