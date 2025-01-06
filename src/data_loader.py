# File: src/data_loader.py
import pandas as pd  # For handling .csv files
import anndata as ad  # For handling .h5ad files
from tqdm import tqdm  # For progress tracking
from typing import Any, Dict, List, Union, Generator  # For type annotations
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from src.loaders.h5ad_loader import H5ADLoader  # Loader for .h5ad files
from src.loaders.tiff_loader import TIFFLoader  # Loader for .tiff files
from src.loaders.zarr_loader import ZARRLoader  # Loader for .zarr files
from src.loaders.csv_loader import CSVLoader  # Loader for .csv files
from src.utils.config_loader import ConfigLoader  # Configuration loader
from src.utils.path_validator import PathValidator  # Path validation utility


class DataLoader:
    """
    A class for managing the loading of various data formats required for the Autoimmune Disease Machine Learning Challenge.

    Features:
    - Handles single or directory-based Zarr files intelligently.
    - Batch and parallel processing for faster loading.
    - Streaming support for `.csv` and `.h5ad` files.
    - Configurable defaults for maximum workers, batch sizes, and streaming chunk sizes.
    """

    ### 1. Initialization and Validation ###
    def __init__(self, config: ConfigLoader, crunch_name: str, max_workers: int = None, batch_size: int = None):
        """
        Initialize the DataLoader with configuration settings and path validation.

        Args:
            config (ConfigLoader): Configuration loader instance.
            crunch_name (str): Name of the Crunch (e.g., "crunch1", "crunch2").
            max_workers (int): Number of parallel workers for processing (default: 4).
            batch_size (int): Number of datasets to process in each batch (default: 2).
        """
        # Save the provided configuration loader instance
        self.config = config

        # Save the name of the Crunch (e.g., "crunch1")
        self.crunch_name = crunch_name

        # Initialize PathValidator for validating paths based on configuration
        self.path_validator = PathValidator(config_path=config.config_path)

        # Set the maximum number of workers for parallel processing, default to config value or 4
        self.max_workers = max_workers or self.config.get_global_setting("max_workers", 4)

        # Set the batch size for processing datasets, default to config value or 2
        self.batch_size = batch_size or self.config.get_global_setting("batch_size", 2)

    def validate_path(self, key: str, is_file: bool = True) -> str:
        """
        Validate a path using PathValidator and return the validated path.

        Args:
            key (str): The key in the configuration file for the path.
            is_file (bool): Whether the path is expected to be a file.

        Returns:
            str: The validated path.

        Raises:
            FileNotFoundError: If the path is not defined or invalid.
        """
        # Retrieve the path for the specified key from the configuration
        path = self.config.get_crunch_path(self.crunch_name, key)

        # Check if the path is None and log an appropriate error
        if not path:
            error_message = f"Path Key '{key}' not found in configuration for '{self.crunch_name}'"
            self.path_validator.logger.error(error_message)
            raise FileNotFoundError(error_message)

        # Validate the path using PathValidator
        try:
            self.path_validator.ensure_path(path, is_file=is_file)
        except FileNotFoundError as e:
            # Log the error if path validation fails
            self.path_validator.logger.error(f"Validation failed for path: {path}. Error: {e}")
            raise

        # Return the validated path
        return path

    ### 2. Loading Individual Datasets ###
    def load_batch(self, keys: List[str], loader_class: Any) -> Dict[str, Any]:
        """
        Load a batch of datasets in parallel.

        Args:
            keys (List[str]): List of dataset keys to load.
            loader_class (Any): Loader class to use.

        Returns:
            Dict[str, Any]: Dictionary of loaded datasets with keys as dataset names.
        """
        # Initialize a dictionary to store the results of the batch loading
        results = {}

        # Use ThreadPoolExecutor to load datasets in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit loading tasks to the executor for each key
            futures = {executor.submit(self._load_with_loader, key, loader_class): key for key in keys}

            # Process completed futures using tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batch"):
                key = futures[future]
                try:
                    # Store the result of the successfully loaded dataset
                    results[key] = future.result()
                except Exception as e:
                    # Log any errors that occur during loading
                    self.path_validator.logger.error(f"Error loading {key}: {e}")

        # Return the dictionary of loaded datasets
        return results

    def _load_with_loader(self, key: str, loader_class: Any) -> Any:
        """
        Load a single dataset using the specified loader.

        Args:
            key (str): Dataset key in the configuration.
            loader_class (Any): Loader class to use.

        Returns:
            Any: Loaded dataset.
        """
        # Validate the file path for the dataset
        path = self.validate_path(key, is_file=True)

        # Use the specified loader to load the dataset
        return loader_class(path).load()

    ### 3. Zarr-Specific Loading ###
    def load_zarr(self, keys: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Load Zarr files or directories containing multiple Zarr datasets.

        Args:
            keys (Union[str, List[str]]): A single Zarr key or a list of keys to load.

        Returns:
            Dict[str, Any]: A dictionary of Zarr datasets, keyed by their names.
        """
        # Ensure `keys` is a list for consistent processing
        if isinstance(keys, str):
            keys = [keys]

        # Initialize a dictionary to store loaded Zarr datasets
        zarr_datasets = {}

        # Use ThreadPoolExecutor for parallel processing of Zarr files
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for loading each Zarr key
            futures = {executor.submit(self._load_with_loader, key, ZARRLoader): key for key in keys}

            # Process completed futures using tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Zarr files"):
                key = futures[future]
                try:
                    # Store the result of the successfully loaded Zarr dataset
                    zarr_datasets[key] = future.result()
                except Exception as e:
                    # Log any errors that occur during loading
                    self.path_validator.logger.error(f"Error loading Zarr file {key}: {e}")

        # Return the dictionary of loaded Zarr datasets
        return zarr_datasets

    ### 4. Streaming for Large Datasets ###
    def stream_csv(self, key: str, chunk_size: int = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a .csv file in chunks.

        Args:
            key (str): The key in the configuration file for the .csv file path.
            chunk_size (int): Number of rows to load per chunk (default: 1000).

        Yields:
            pandas.DataFrame: A chunk of the loaded CSV data.
        """
        # Validate the file path for the .csv file
        path = self.validate_path(key, is_file=True)

        # Set the chunk size, default to the global configuration setting
        chunk_size = chunk_size or self.config.get_global_setting("csv_chunk_size", 1000)

        # Stream chunks of the .csv file
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            yield chunk

    def stream_h5ad(self, key: str, chunk_size: int = None) -> Generator[ad.AnnData, None, None]:
        """
        Stream a .h5ad file in chunks.

        Args:
            key (str): The key in the configuration file for the .h5ad file path.
            chunk_size (int): Number of rows to load per chunk (default: 1000).

        Yields:
            anndata.AnnData: A chunk of the loaded AnnData object.
        """
        # Validate the file path for the .h5ad file
        path = self.validate_path(key, is_file=True)

        # Load the entire .h5ad dataset
        adata = ad.read_h5ad(path)

        # Set the chunk size, default to the global configuration setting
        chunk_size = chunk_size or self.config.get_global_setting("h5ad_chunk_size", 1000)

        # Stream chunks of the dataset
        for i in range(0, adata.shape[0], chunk_size):
            yield adata[i : i + chunk_size]

    ### 5. Orchestrated Loading ###
    def load_all(self, streaming: Union[bool, List[str]] = False) -> Dict[str, Any]:
        """
        Validate and load all datasets for the specified Crunch.

        Args:
            streaming (Union[bool, List[str]]): Whether to enable streaming for large datasets,
                                                or a list of file types to stream.

        Returns:
            Dict[str, Any]: A dictionary with loaded datasets or streaming generators.
        """
        # Initialize a dictionary to store loaded datasets and errors
        datasets = {}
        errors = []

        # Define dataset types and their associated keys and loaders
        data_types = {
            "h5ad_files": (["scRNA_seq_file"], H5ADLoader),
            "tiff_files": (["he_image_file", "he_label_file", "he_dysplasia_roi_file"], TIFFLoader),
            "zarr_files": (["raw_dir"], ZARRLoader),
            "csv_files": (["gene_list_file"], CSVLoader),
        }

        # Iterate through each data type
        for data_type, (keys, loader_class) in data_types.items():
            if isinstance(streaming, list) and data_type in streaming:
                # Enable streaming for specific file types
                datasets[f"{data_type}_streams"] = {
                    key: self.stream_csv(key) if data_type == "csv_files" else self.stream_h5ad(key)
                    for key in keys
                }
            elif streaming is True and data_type in ["csv_files", "h5ad_files"]:
                # Enable streaming for all large file types
                datasets[f"{data_type}_streams"] = {
                    key: self.stream_csv(key) if data_type == "csv_files" else self.stream_h5ad(key)
                    for key in keys
                }
            else:
                try:
                    # Load datasets fully for non-streaming cases
                    datasets[data_type] = self.load_batch(keys, loader_class)
                except Exception as e:
                    # Collect errors for failed dataset loads
                    errors.append(f"Failed to load {data_type}: {e}")

        # Log all errors encountered during dataset loading
        if errors:
            self.path_validator.logger.error(f"Errors encountered: {errors}")

        # Return the dictionary of loaded datasets
        return datasets
