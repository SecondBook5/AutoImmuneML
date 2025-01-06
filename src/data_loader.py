import pandas as pd  # For handling .csv files
import anndata as ad  # For handling .h5ad files
from tqdm import tqdm  # For progress tracking
from typing import Any, Dict, List, Union, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from src.loaders.h5ad_loader import H5ADLoader
from src.loaders.tiff_loader import TIFFLoader
from src.loaders.zarr_loader import ZARRLoader
from src.loaders.csv_loader import CSVLoader
from src.utils.config_loader import ConfigLoader  # For loading configuration settings
from src.utils.path_validator import PathValidator  # For validating paths


class DataLoader:
    """
    A class for managing the loading of various data formats required for the Autoimmune Disease Machine Learning Challenge.

    Features:
    - Supports individual file-type loading (e.g., Zarr-specific loading with `load_zarr`).
    - Batch and parallel processing for faster loading.
    - Streaming support for `.csv` and `.h5ad` files.
    """

    def __init__(self, config: ConfigLoader, crunch_name: str, max_workers: int = 4, batch_size: int = 2):
        """
        Initialize the DataLoader with configuration settings and path validation.

        Args:
            config (ConfigLoader): Configuration loader instance.
            crunch_name (str): Name of the Crunch (e.g., "crunch1", "crunch2").
            max_workers (int): Number of parallel workers for processing (default: 4).
            batch_size (int): Number of datasets to process in each batch (default: 2).
        """
        self.config = config  # Configuration loader instance
        self.crunch_name = crunch_name  # Name of the Crunch (e.g., "crunch1")
        self.path_validator = PathValidator(config_path=config.config_path)  # Initialize PathValidator
        self.max_workers = max_workers  # Define the number of parallel workers
        self.batch_size = batch_size  # Define the batch size for processing

    def validate_path(self, key: str, is_file: bool = True) -> str:
        """
        Validate a path using PathValidator and return the validated path.

        Args:
            key (str): The key in the configuration file for the path.
            is_file (bool): Whether the path is expected to be a file.

        Returns:
            str: The validated path.
        """
        path = self.config.get_crunch_path(self.crunch_name, key)  # Retrieve the path from the configuration
        self.path_validator.ensure_path(path, is_file=is_file)  # Validate the path
        return path  # Return the validated path

    def _load_with_loader(self, key: str, loader_class: Any) -> Any:
        """
        Load a single dataset using the specified loader.

        Args:
            key (str): Dataset key in the configuration.
            loader_class (Any): Loader class to use.

        Returns:
            Any: Loaded dataset.
        """
        path = self.validate_path(key, is_file=True)  # Validate the path for the dataset
        return loader_class(path).load()  # Use the specified loader to load the dataset

    def load_batch(self, keys: List[str], loader_class: Any) -> Dict[str, Any]:
        """
        Load a batch of datasets in parallel.

        Args:
            keys (List[str]): List of dataset keys to load.
            loader_class (Any): Loader class to use.

        Returns:
            Dict[str, Any]: Dictionary of loaded datasets with keys as dataset names.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_with_loader, key, loader_class): key for key in keys}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batch"):
                key = futures[future]
                try:
                    results[key] = future.result()  # Store the result
                except Exception as e:
                    self.path_validator.logger.error(f"Error loading {key}: {e}")  # Log any errors
        return results

    def stream_csv(self, key: str, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a .csv file in chunks.

        Args:
            key (str): The key in the configuration file for the .csv file path.
            chunk_size (int): Number of rows to load per chunk.

        Yields:
            pandas.DataFrame: A chunk of the loaded CSV data.
        """
        path = self.validate_path(key, is_file=True)  # Validate the .csv file path
        for chunk in pd.read_csv(path, chunksize=chunk_size):  # Read and yield chunks
            yield chunk

    def stream_h5ad(self, key: str, chunk_size: int = 1000) -> Generator[ad.AnnData, None, None]:
        """
        Stream a .h5ad file in chunks.

        Args:
            key (str): The key in the configuration file for the .h5ad file path.
            chunk_size (int): Number of rows to load per chunk.

        Yields:
            anndata.AnnData: A chunk of the loaded AnnData object.
        """
        path = self.validate_path(key, is_file=True)  # Validate the .h5ad file path
        adata = ad.read_h5ad(path)  # Load the full .h5ad dataset
        for i in range(0, adata.shape[0], chunk_size):  # Yield chunks of the dataset
            yield adata[i : i + chunk_size]

    def load_all(self, streaming: bool = False) -> Dict[str, Any]:
        """
        Validate and load all datasets for the specified Crunch.

        Args:
            streaming (bool): Whether to enable streaming for large datasets.

        Returns:
            Dict[str, Any]: A dictionary with loaded datasets or streaming generators.
        """
        datasets = {}  # Initialize an empty dictionary to store datasets
        h5ad_keys = ["scRNA_seq_file"]
        tiff_keys = ["he_image_file", "he_label_file", "he_dysplasia_roi_file"]
        zarr_keys = ["raw_dir", "interim_dir"]
        csv_keys = ["gene_list_file"]

        if streaming:
            # Return streaming generators for .csv and .h5ad files
            datasets["h5ad_streams"] = {key: self.stream_h5ad(key) for key in h5ad_keys}
            datasets["csv_streams"] = {key: self.stream_csv(key) for key in csv_keys}
        else:
            # Load datasets fully for all formats
            datasets["h5ad_files"] = self.load_batch(h5ad_keys, H5ADLoader)
            datasets["tiff_files"] = self.load_batch(tiff_keys, TIFFLoader)
            datasets["zarr_files"] = self.load_batch(zarr_keys, ZARRLoader)
            datasets["csv_files"] = self.load_batch(csv_keys, CSVLoader)

        return datasets
