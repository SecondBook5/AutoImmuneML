# File: src/validators/spatialdata_validator.py

import os
import spatialdata as sd
import tifffile as tiff
import pandas as pd
import logging
from typing import Dict, List
from src.validators.validation_logger import ValidationLogger
from src.utils.manifest_manager import ManifestManager


class SpatialDataValidator:
    """
    Validates spatial data files, including .zarr, .tiff, .h5ad, and .csv formats.
    Integrates spatialdata, squidpy, and scanpy for robust validation.
    """

    def __init__(self, base_dir: str, manifest_path: str):
        """
        Initialize the validator with the base directory and manifest path.

        Args:
            base_dir (str): Base directory containing spatial data.
            manifest_path (str): Path to the manifest file.
        """
        self.base_dir = base_dir  # Directory where spatial data resides
        self.logger = ValidationLogger(manifest_path)  # Logger for validation results
        self.manifest_manager = ManifestManager(manifest_path)  # Manages manifest loading and saving

    def validate_tiff(self, tiff_path: str, crunch_name: str):
        """
        Validate the integrity of a .tiff file and log the result.

        Args:
            tiff_path (str): Path to the .tiff file.
            crunch_name (str): Name of the Crunch.
        """
        try:
            # Attempt to open and read the TIFF file to validate its integrity
            with tiff.TiffFile(tiff_path) as tif:
                tif.asarray()  # Attempt to read the TIFF file to check integrity
            self.logger.log(crunch_name, tiff_path, "valid", {"type": "tiff"})
        except Exception as e:
            self.logger.log(crunch_name, tiff_path, "invalid", {"type": "tiff", "error": str(e)})

    def validate_zarr(self, zarr_path: str, crunch_name: str):
        """
        Validate a .zarr file using spatialdata and log the result.

        Args:
            zarr_path (str): Path to the .zarr file.
            crunch_name (str): Name of the Crunch.
        """
        try:
            sd.read_zarr(zarr_path)  # Validate the structure of the Zarr file
            self.logger.log(crunch_name, zarr_path, "valid", {"type": "zarr"})
        except Exception as e:
            self.logger.log(crunch_name, zarr_path, "invalid", {"type": "zarr", "error": str(e)})

    def validate_h5ad(self, h5ad_path: str, crunch_name: str):
        """
        Validate an .h5ad file using squidpy and log the result.

        Args:
            h5ad_path (str): Path to the .h5ad file.
            crunch_name (str): Name of the Crunch.
        """
        try:
            import anndata
            import squidpy as sq
            adata = anndata.read_h5ad(h5ad_path)
            sq.gr.spatial_neighbors(adata)  # Validate spatial neighbors
            self.logger.log(crunch_name, h5ad_path, "valid", {"type": "h5ad"})
        except Exception as e:
            self.logger.log(crunch_name, h5ad_path, "invalid", {"type": "h5ad", "error": str(e)})

    def validate_csv(self, csv_path: str, crunch_name: str):
        """
        Validate a .csv file for expected structure and log the result.

        Args:
            csv_path (str): Path to the .csv file.
            crunch_name (str): Name of the Crunch.
        """
        try:
            data = pd.read_csv(csv_path)
            if {"Gene", "Expression"}.issubset(data.columns):  # Check for required columns
                self.logger.log(crunch_name, csv_path, "valid", {"type": "csv"})
            else:
                raise ValueError("Missing required columns: 'Gene', 'Expression'.")
        except Exception as e:
            self.logger.log(crunch_name, csv_path, "invalid", {"type": "csv", "error": str(e)})

    def run_validation(self):
        """
        Perform validation for all files listed in the manifest.
        """
        manifest = self.manifest_manager.load_manifest()

        for crunch_name, files in manifest.items():
            print(f"Validating Crunch: {crunch_name}")
            for file_path in files:
                full_path = os.path.join(self.base_dir, file_path)

                if file_path.endswith(".zarr"):
                    self.validate_zarr(full_path, crunch_name)
                elif file_path.endswith(".tiff"):
                    self.validate_tiff(full_path, crunch_name)
                elif file_path.endswith(".h5ad"):
                    self.validate_h5ad(full_path, crunch_name)
                elif file_path.endswith(".csv"):
                    self.validate_csv(full_path, crunch_name)
                else:
                    self.logger.log(crunch_name, file_path, "unsupported", {"type": "unknown"})
                    print(f"Unsupported file type: {file_path}")

        print("Validation process complete.")
