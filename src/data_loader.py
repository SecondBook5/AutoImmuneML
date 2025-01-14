# File: src/data_loader.py
import os
import pandas as pd  # For handling .csv files
import anndata as ad  # For handling .h5ad files
from tqdm import tqdm  # For progress tracking
from typing import Any, Dict, List, Union, Generator  # For type annotations
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
from src.loaders.h5ad_loader import H5ADLoader  # Loader for .h5ad files
from src.loaders.tiff_loader import TIFFLoader  # Loader for .tiff files
from src.loaders.zarr_loader import ZARRLoader  # Loader for .zarr files
from src.loaders.csv_loader import CSVLoader  # Loader for .csv files
from src.config.config_loader import ConfigLoader  # Configuration loader
from src.validators.zarr_validator import ZARRValidator  # Validator for .zarr datasets


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
        Initialize the DataLoader with configuration settings.

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
        # Set the maximum number of workers for parallel processing, default to config value or 4
        self.max_workers = max_workers or self.config.get_global_setting("max_workers", 4)
        # Set the batch size for processing datasets, default to config value or 2
        self.batch_size = batch_size or self.config.get_global_setting("batch_size", 2)

    ### 2. Zarr Validation and Loading ###
    def validate_zarr(self, path: str) -> bool:
        """
        Validate a `.zarr` dataset using ZARRValidator.

        Args:
            path (str): Path to a `.zarr` dataset or directory.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        validator = ZARRValidator(path)
        result = validator.validate()
        if result["status"] == "valid":
            return True
        else:
            print(f"[ERROR] Validation failed for {path}: {result['errors']}")
            return False

    def load_zarr(self, paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Load Zarr files or directories containing multiple Zarr datasets.

        Args:
            paths (Union[str, List[str]]): A single Zarr path or a list of paths.

        Returns:
            Dict[str, Any]: A dictionary of Zarr datasets, keyed by their names.
        """
        # Ensure paths is always a list for consistent processing
        if isinstance(paths, str):
            paths = [paths]

        # Initialize a dictionary to store loaded Zarr datasets
        zarr_datasets = {}

        # Initialize a list to hold all Zarr paths to process
        zarr_paths_to_load = []

        # Iterate over the provided paths and validate them
        for path in paths:
            # Check if the path exists
            if not os.path.exists(path):
                print(f"[ERROR] Path does not exist: {path}")
                continue
            # If the path is a single `.zarr` directory, add it to the list
            if os.path.isdir(path) and path.endswith(".zarr"):
                zarr_paths_to_load.append(path)
            # If the path is a directory, add all `.zarr` directories within it
            elif os.path.isdir(path):
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path) and item_path.endswith(".zarr"):
                        zarr_paths_to_load.append(item_path)

        # Define the function to load a single `.zarr` dataset
        def load_single_zarr(zarr_path):
            try:
                # Validate the Zarr dataset structure
                if self.validate_zarr(zarr_path):
                    # Load and return the dataset if validation passes
                    return os.path.basename(zarr_path), ZARRLoader(zarr_path).load()
                else:
                    # If validation fails, return None for the dataset
                    return os.path.basename(zarr_path), None
            except Exception as e:
                # Log any exceptions encountered during loading
                print(f"[ERROR] Failed to load Zarr dataset {zarr_path}: {e}")
                # Return None for failed datasets
                return os.path.basename(zarr_path), None

        # Use ThreadPoolExecutor to load Zarr datasets in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all Zarr paths for loading and track the futures
            futures = {executor.submit(load_single_zarr, path): path for path in zarr_paths_to_load}
            # Process the completed futures with a progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Zarr files"):
                # Retrieve the dataset name and data from the future result
                dataset_name, data = future.result()
                # If the dataset was loaded successfully, store it in the dictionary
                if data is not None:
                    zarr_datasets[dataset_name] = data

        # Return the dictionary containing all loaded Zarr datasets
        return zarr_datasets

    ### 3. Batch Loading ###
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
            futures = {executor.submit(loader_class(key).load): key for key in keys}
            # Process completed futures using tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batch"):
                key = futures[future]
                try:
                    # Store the result of the successfully loaded dataset
                    results[key] = future.result()
                except Exception as e:
                    print(f"[ERROR] Error loading {key}: {e}")
        # Return the dictionary of loaded datasets
        return results

    ### 4. Streaming Support ###
    def stream_csv(self, path: str, chunk_size: int = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a .csv file in chunks.

        Args:
            path (str): Path to the .csv file.
            chunk_size (int): Number of rows to load per chunk (default: 1000).

        Yields:
            pandas.DataFrame: A chunk of the loaded CSV data.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found at {path}")

        # Set the chunk size, default to the global configuration setting
        chunk_size = chunk_size or self.config.get_global_setting("csv_chunk_size", 1000)

        # Stream chunks of the .csv file
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            yield chunk

    def stream_h5ad(self, path: str, chunk_size: int = None) -> Generator[ad.AnnData, None, None]:
        """
        Stream a .h5ad file in chunks.

        Args:
            path (str): Path to the .h5ad file.
            chunk_size (int): Number of rows to load per chunk (default: 1000).

        Yields:
            anndata.AnnData: A chunk of the loaded AnnData object.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"H5AD file not found at {path}")

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
                datasets[data_type] = self.load_batch(keys, loader_class)
        return datasets
