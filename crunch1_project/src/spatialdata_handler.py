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

    def load_data(self) -> None:
        """
        Load all SpatialData objects from the provided Zarr files using multithreading.

        Raises:
            FileNotFoundError: If a specified Zarr file is not found.
            Exception: If any dataset fails to load, details are logged.
        """
        # Load all SpatialData objects from the specified Zarr files using multithreading

        def load_single_dataset(zarr_path: str) -> None:
            # Load a single SpatialData object from a Zarr file
            if not os.path.exists(zarr_path):  # Check if the file exists
                raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
            try:
                dataset_name = os.path.basename(zarr_path)  # Extract dataset name from path
                print(f"Loading dataset: {zarr_path}...")  # Log loading start
                dataset = sd.read_zarr(zarr_path)  # Load the SpatialData object
                self.datasets[dataset_name] = dataset  # Store the dataset in the dictionary
                print(f"Dataset '{dataset_name}' loaded successfully.")  # Log success
            except Exception as e:
                print(f"Failed to load dataset '{zarr_path}': {e}")  # Log errors
                raise

        errors = []  # List to keep track of errors during dataset loading
        with ThreadPoolExecutor() as executor:
            # Submit all dataset loading tasks to a thread pool for parallel processing
            futures = {executor.submit(load_single_dataset, path): path for path in self.zarr_paths}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.zarr_paths), desc="Loading Datasets"):
                try:
                    future.result()  # Wait for the task to complete
                except Exception as e:
                    errors.append((futures[future], e))  # Record errors

        if errors:  # Log errors for failed datasets
            print("Some datasets failed to load:")
            for path, error in errors:
                print(f" - {path}: {error}")

    def validate_data(self, required_images: Optional[List[str]] = None, required_tables: Optional[List[str]] = None) -> None:
        """
        Validate the presence of required images and tables in all loaded datasets.

        Args:
            required_images (Optional[List[str]]): A list of required image keys.
            required_tables (Optional[List[str]]): A list of required table keys.

        Raises:
            ValueError: If a required component (image/table) is missing in any dataset.
        """
        # Validate all loaded datasets for the required components
        required_images = required_images or ["HE_original", "HE_nuc_original"]  # Default image keys
        required_tables = required_tables or ["anucleus"]  # Default table keys

        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Validating Datasets"):
            # Iterate over each loaded dataset
            print(f"Validating dataset: {dataset_name}...")  # Log validation start
            missing_components: List[str] = []  # List of missing components

            for key in required_images:  # Check required images
                if key not in dataset.images.keys():  # Add missing image keys to the list
                    missing_components.append(f"Image '{key}'")

            for key in required_tables:  # Check required tables
                if key not in dataset.tables.keys():  # Add missing table keys to the list
                    missing_components.append(f"Table '{key}'")

            if missing_components:  # Log missing components
                print(f"Dataset '{dataset_name}' is missing the following components:")
                for component in missing_components:
                    print(f" - {component}")
            else:
                print(f"Dataset '{dataset_name}' is fully validated.")  # Log validation success

    def subsample_data(self, max_cells: int = 1000) -> None:
        """
        Subsample all datasets to limit the number of cells for memory efficiency.

        Args:
            max_cells (int): Maximum number of cells to retain in each dataset.
        """
        # Subsample all loaded datasets to limit the number of cells
        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Subsampling Datasets"):
            # Iterate over each dataset for subsampling
            anucleus = dataset.tables.get("anucleus")  # Access the anucleus table
            if anucleus is None:  # Skip if the table is missing
                print(f"Dataset '{dataset_name}' does not contain an 'anucleus' table. Skipping subsampling.")
                continue

            total_cells = anucleus.shape[0]  # Get the total number of cells
            if total_cells > max_cells:  # Check if subsampling is required
                print(f"Subsampling {total_cells} cells to {max_cells} in dataset '{dataset_name}'...")
                subsample_indices = np.random.choice(anucleus.obs_names, max_cells, replace=False)  # Select random cells
                self.datasets[dataset_name].tables["anucleus"] = anucleus[subsample_indices, :]  # Update the table
                print(f"Subsampled dataset '{dataset_name}' to {max_cells} cells.")  # Log success
            else:
                print(f"No subsampling needed for dataset '{dataset_name}'.")  # Log if no subsampling is required

    def extract_nuclei_and_gene_expression(self, gene_subset: Optional[List[str]] = None, nuclei_subset: Optional[List[str]] = None,) -> Dict[str, pd.DataFrame]:
        """
        Extract nuclei coordinates and link them to their gene expression data.

        Args:
            gene_subset (Optional[List[str]]): A subset of genes to extract.
            nuclei_subset (Optional[List[str]]): A subset of nuclei to include.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping dataset names to their corresponding dataframes.
        """
        # Extract nuclei coordinates and link them to gene expression data for all datasets
        results: Dict[str, pd.DataFrame] = {}  # Initialize results dictionary

        for dataset_name, dataset in tqdm(self.datasets.items(), desc="Extracting Data"):
            # Iterate over each dataset for data extraction
            anucleus = dataset.tables.get("anucleus")  # Get the anucleus table
            if anucleus is None:  # Skip if the table is missing
                print(f"Dataset '{dataset_name}' does not contain an 'anucleus' table. Skipping extraction.")
                continue

            nuclei_coords = anucleus.obsm["spatial"]  # Extract spatial coordinates
            nuclei_ids = anucleus.obs_names  # Extract nuclei IDs
            gene_expression = anucleus.X  # Extract gene expression data
            gene_names = anucleus.var_names  # Extract gene names

            # Apply nuclei and gene subsetting if specified
            if nuclei_subset:
                nuclei_mask = np.isin(nuclei_ids, nuclei_subset)
                nuclei_coords = nuclei_coords[nuclei_mask]
                nuclei_ids = nuclei_ids[nuclei_mask]

            if gene_subset:
                gene_mask = np.isin(gene_names, gene_subset)
                gene_expression = gene_expression[:, gene_mask]
                gene_names = gene_names[gene_mask]

            # Create a DataFrame for the dataset
            data: Dict[str, List] = {
                "nuclei_id": nuclei_ids,
                "x_coord": nuclei_coords[:, 0],
                "y_coord": nuclei_coords[:, 1],
            }

            for i, gene in tqdm(enumerate(gene_names), desc=f"Processing Genes in {dataset_name}", total=len(gene_names)):
                # Populate the DataFrame with gene expression data
                data[gene] = gene_expression[:, i]

            results[dataset_name] = pd.DataFrame(data)  # Add results to the dictionary
            print(f"Extraction complete for dataset '{dataset_name}'.")  # Log success

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
