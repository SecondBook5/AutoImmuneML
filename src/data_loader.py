import pandas as pd  # Import pandas for handling .csv files
import anndata as ad  # Import anndata for handling .h5ad files
from tqdm import tqdm  # Import tqdm for progress tracking
from typing import Any, Dict, List, Union, Generator  # Import type annotations
from concurrent.futures import ThreadPoolExecutor, as_completed  # Import for parallel processing
from src.loaders.h5ad_loader import H5ADLoader  # Import loader for .h5ad files
from src.loaders.tiff_loader import TIFFLoader  # Import loader for .tiff files
from src.loaders.zarr_loader import ZARRLoader  # Import loader for .zarr files
from src.loaders.csv_loader import CSVLoader  # Import loader for .csv files
from src.utils.config_loader import ConfigLoader  # Import configuration loader
from src.utils.path_validator import PathValidator  # Import path validator


class DataLoader:
    """
    A class for managing the loading of various data formats required for the Autoimmune Disease Machine Learning Challenge.

    Features:
    - Handles single or directory-based Zarr files intelligently.
    - Batch and parallel processing for faster loading.
    - Streaming support for `.csv` and `.h5ad` files.
    """

    ### 1. Initialization and Validation ###
    def __init__(self, config: ConfigLoader, crunch_name: str, max_workers: int = 4, batch_size: int = 2):
        """
        Initialize the DataLoader with configuration settings and path validation.

        Args:
            config (ConfigLoader): Configuration loader instance.
            crunch_name (str): Name of the Crunch (e.g., "crunch1", "crunch2").
            max_workers (int): Number of parallel workers for processing (default: 4).
            batch_size (int): Number of datasets to process in each batch (default: 2).
        """
        # Store the configuration loader instance
        self.config = config

        # Store the Crunch name
        self.crunch_name = crunch_name

        # Initialize PathValidator to ensure all paths are valid
        self.path_validator = PathValidator(config_path=config.config_path)

        # Define the maximum number of workers for parallel processing
        self.max_workers = max_workers

        # Define the batch size for processing
        self.batch_size = batch_size

    def validate_path(self, key: str, is_file: bool = True) -> str:
        """
        Validate a path using PathValidator and return the validated path.

        Args:
            key (str): The key in the configuration file for the path.
            is_file (bool): Whether the path is expected to be a file.

        Returns:
            str: The validated path.
        """
        # Retrieve the path from the configuration using the Crunch name and key
        path = self.config.get_crunch_path(self.crunch_name, key)

        # Validate the path using PathValidator
        self.path_validator.ensure_path(path, is_file=is_file)

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
        # Initialize an empty dictionary to store the results
        results = {}

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each key to the executor and track their futures
            futures = {executor.submit(self._load_with_loader, key, loader_class): key for key in keys}

            # Process each future as it completes
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading batch"):
                key = futures[future]
                try:
                    # Store the result of the loaded dataset
                    results[key] = future.result()
                except Exception as e:
                    # Log any errors encountered during loading
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
        # Ensure `keys` is a list for uniform processing
        if isinstance(keys, str):
            keys = [keys]

        # Initialize a dictionary to store loaded Zarr datasets
        zarr_datasets = {}

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each Zarr key and track their futures
            futures = {executor.submit(self._load_with_loader, key, ZARRLoader): key for key in keys}

            # Process each future as it completes
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Zarr files"):
                key = futures[future]
                try:
                    # Store the loaded Zarr dataset
                    zarr_datasets[key] = future.result()
                except Exception as e:
                    # Log any errors encountered during loading
                    self.path_validator.logger.error(f"Error loading Zarr file {key}: {e}")

        # Return the dictionary of loaded Zarr datasets
        return zarr_datasets

    ### 4. Streaming for Large Datasets ###
    def stream_csv(self, key: str, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a .csv file in chunks.

        Args:
            key (str): The key in the configuration file for the .csv file path.
            chunk_size (int): Number of rows to load per chunk.

        Yields:
            pandas.DataFrame: A chunk of the loaded CSV data.
        """
        # Validate the .csv file path
        path = self.validate_path(key, is_file=True)

        # Use pandas to read the .csv file in chunks and yield each chunk
        for chunk in pd.read_csv(path, chunksize=chunk_size):
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
        # Validate the .h5ad file path
        path = self.validate_path(key, is_file=True)

        # Load the full .h5ad dataset
        adata = ad.read_h5ad(path)

        # Yield chunks of the dataset
        for i in range(0, adata.shape[0], chunk_size):
            yield adata[i : i + chunk_size]

    ### 5. Orchestrated Loading ###
    def load_all(self, streaming: bool = False) -> Dict[str, Any]:
        """
        Validate and load all datasets for the specified Crunch.

        Args:
            streaming (bool): Whether to enable streaming for large datasets.

        Returns:
            Dict[str, Any]: A dictionary with loaded datasets or streaming generators.
        """
        # Initialize an empty dictionary to store datasets
        datasets = {}

        # Define dataset keys for each file type
        h5ad_keys = ["scRNA_seq_file"]
        tiff_keys = ["he_image_file", "he_label_file", "he_dysplasia_roi_file"]
        zarr_keys = ["raw_dir", "interim_dir"]
        csv_keys = ["gene_list_file"]

        if streaming:
            # Return streaming generators for .csv and .h5ad files
            datasets["h5ad_streams"] = {key: self.stream_h5ad(key) for key in h5ad_keys}
            datasets["csv_streams"] = {key: self.stream_csv(key) for key in csv_keys}
        else:
            # Fully load datasets for all formats
            datasets["h5ad_files"] = self.load_batch(h5ad_keys, H5ADLoader)
            datasets["tiff_files"] = self.load_batch(tiff_keys, TIFFLoader)
            datasets["zarr_files"] = self.load_zarr(zarr_keys)
            datasets["csv_files"] = self.load_batch(csv_keys, CSVLoader)

        # Return the dictionary of loaded datasets
        return datasets
