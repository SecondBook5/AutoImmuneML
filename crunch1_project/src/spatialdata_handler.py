# File: crunch1_project/src/spatialdata_handler.py

import spatialdata as sd  # For handling spatial omics datasets
import matplotlib.pyplot as plt  # For static 2D visualizations
import plotly.express as px  # For interactive visualizations
import os  # For file and path operations
import logging  # For logging and debugging
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and storage
from typing import Optional  # For type annotations

# Configure the logging system to capture and display information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Create a logger instance for the class


class SpatialDataHandler:
    """
    A class to manage loading, validating, visualizing, and processing SpatialData objects.
    """

    def __init__(self, zarr_path: str):
        """
        Initialize the SpatialDataHandler with the path to a Zarr dataset.

        Args:
            zarr_path (str): Path to the Zarr file containing the dataset.
        """
        self.zarr_path = zarr_path  # Save the Zarr file path as an instance variable
        self.sdata = None  # Placeholder for the loaded SpatialData object
        logger.info(f"SpatialDataHandler initialized for file: {zarr_path}")  # Log initialization

    def load_data(self) -> None:
        """
        Load the SpatialData object from the Zarr file.

        Raises:
            FileNotFoundError: If the Zarr file does not exist.
        """
        # Check if the file exists before attempting to load
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")

        try:
            # Use spatialdata to load the Zarr dataset
            self.sdata = sd.read_zarr(self.zarr_path)
            logger.info("SpatialData loaded successfully.")  # Log success
            logger.info(f"Loaded SpatialData Summary:\n{self.sdata}")  # Log dataset details
        except Exception as e:
            # Log and raise errors if the loading process fails
            logger.error(f"Failed to load SpatialData: {e}")
            raise

    def validate_data(self) -> None:
        """
        Validate that the loaded SpatialData object contains the required components.

        Raises:
            ValueError: If required images or tables are missing.
        """
        # Ensure data is loaded before validating
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Define the required components for the dataset
        required_images = ["HE_original", "HE_nuc_original"]
        required_tables = ["anucleus"]

        # Check if the required image keys are present
        for key in required_images:
            if key not in self.sdata.images.keys():
                raise ValueError(f"Missing required image: {key}")

        # Check if the required table keys are present
        for key in required_tables:
            if key not in self.sdata.tables.keys():
                raise ValueError(f"Missing required table: {key}")

        logger.info("SpatialData validation passed.")  # Log successful validation

    def subsample_data(self, max_cells: int = 1000) -> None:
        """
        Subsample the SpatialData object to limit the number of cells for memory efficiency.

        Args:
            max_cells (int): Maximum number of cells to retain.
        """
        # Ensure data is loaded before subsampling
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Extract the anucleus table for subsampling
        anucleus = self.sdata.tables["anucleus"]
        if anucleus.shape[0] > max_cells:
            # Randomly select indices to subsample the dataset
            subsample_indices = np.random.choice(anucleus.obs_names, max_cells, replace=False)
            # Update the anucleus table with the subsampled data
            self.sdata.tables["anucleus"] = anucleus[subsample_indices, :]
            logger.info(f"Subsampled to {max_cells} cells.")  # Log successful subsampling
        else:
            logger.info("No subsampling needed; dataset size is within limit.")  # Log when no subsampling is necessary

    def visualize_image(self, key: str, cmap: Optional[str] = None) -> None:
        """
        Visualize an image from the SpatialData object using Matplotlib.

        Args:
            key (str): The key of the image to visualize (e.g., "HE_original").
            cmap (Optional[str]): Colormap to use for visualization (e.g., 'gray').

        Raises:
            KeyError: If the specified key is not found in the images.
        """
        # Ensure data is loaded before visualizing
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Check if the image key exists in the SpatialData images
        if key not in self.sdata.images.keys():
            raise KeyError(f"Image '{key}' not found in SpatialData.")

        # Extract the image data and plot it
        image = self.sdata.images[key].to_numpy()
        self._plot_image(image, title=f"Visualizing {key}", cmap=cmap)

    def interactive_visualize(self, key: str) -> None:
        """
        Create an interactive visualization of an image using Plotly.

        Args:
            key (str): The key of the image to visualize (e.g., "HE_original").
        """
        # Ensure data is loaded before creating an interactive visualization
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Check if the image key exists in the SpatialData images
        if key not in self.sdata.images.keys():
            raise KeyError(f"Image '{key}' not found in SpatialData.")

        # Extract the image data and create an interactive Plotly plot
        image = self.sdata.images[key].to_numpy()
        fig = px.imshow(image, title=f"Interactive Visualization: {key}", color_continuous_scale="Viridis")
        fig.update_layout(coloraxis_colorbar={"title": "Intensity"})
        fig.show()

    def extract_nuclei_and_gene_expression(self) -> pd.DataFrame:
        """
        Extract nuclei coordinates and link them to gene expression data.

        Returns:
            pd.DataFrame: A DataFrame containing nuclei coordinates and gene expression data.
        """
        # Ensure data is loaded before extracting nuclei and gene expression
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Extract nuclei coordinates and IDs
        nuclei_coords = self.sdata.tables["anucleus"].obsm["spatial"]
        nuclei_ids = self.sdata.tables["anucleus"].obs_names

        # Extract gene expression data and gene names
        gene_expression = self.sdata.tables["anucleus"].X
        gene_names = self.sdata.tables["anucleus"].var_names

        # Create a DataFrame linking nuclei and gene expression
        data = {
            "nuclei_id": nuclei_ids,
            "x_coord": nuclei_coords[:, 0],
            "y_coord": nuclei_coords[:, 1],
        }
        for i, gene in enumerate(gene_names):
            data[gene] = gene_expression[:, i]

        # Convert the data into a DataFrame
        df = pd.DataFrame(data)
        logger.info(
            f"Extracted data for {df.shape[0]} nuclei and {len(gene_names)} genes.")  # Log successful extraction
        return df

    @staticmethod
    def _plot_image(image: np.ndarray, title: str, cmap: Optional[str] = None) -> None:
        """
        Display a 2D image using Matplotlib.

        Args:
            image (np.ndarray): The image data to display.
            title (str): Title of the plot.
            cmap (Optional[str]): Optional colormap for visualization.
        """
        # Raise an error if the image data is invalid
        if image is None:
            raise ValueError("Cannot plot: Image data is None.")

        # Create and display the plot
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def print_summary(self) -> None:
        """
        Print a summary of the loaded SpatialData object.
        """
        # Ensure data is loaded before printing the summary
        if self.sdata is None:
            raise ValueError("No SpatialData loaded. Please call `load_data` first.")

        # Log and print the SpatialData summary
        logger.info(f"SpatialData Summary:\n{self.sdata}")
        print(self.sdata)
