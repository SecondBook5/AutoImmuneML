import os
import json
import numpy as np
import spatialdata as sd
from typing import Dict, List
import tifffile as tiff  # For validating TIFF file integrity
import pandas as pd  # For handling CSV file validation


class SpatialDataValidator:
    """
    A comprehensive validator for spatial data, including .zarr, .tiff, .csv, and .h5ad files.
    Integrates spatialdata, squidpy, and scanpy for robust validation.
    """

    def __init__(self, base_dir: str, manifest_file: str = "manifest.json"):
        """
        Initialize the validator with the base directory and manifest file.

        Args:
            base_dir (str): Base directory containing spatial data.
            manifest_file (str): Path to the manifest file.
        """
        # Store the base directory where the data resides
        self.base_dir = base_dir
        # Store the path to the manifest file
        self.manifest_file = manifest_file
        # Load the manifest or initialize it if not found
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """
        Load the manifest file.

        Returns:
            Dict: Loaded manifest or an empty dictionary if the file is not found.
        """
        # Check if the manifest file exists
        if not os.path.exists(self.manifest_file):
            print("[✘] Manifest file not found. Starting with an empty manifest.")
            return {}
        # Load and return the manifest content as a dictionary
        with open(self.manifest_file, "r") as f:
            return json.load(f)

    def _save_manifest(self):
        """
        Save the current state of the manifest to the file.
        """
        # Write the manifest data back to the manifest file
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=4)

    def validate_tiff(self, tiff_path: str) -> bool:
        """
        Validate the integrity of a .tiff file.

        Args:
            tiff_path (str): Path to the .tiff file.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Attempt to open and read the TIFF file to validate its integrity
            with tiff.TiffFile(tiff_path) as tif:
                tif.asarray()  # Load the array to check for errors
            return True  # Return True if the file is valid
        except Exception as e:
            # Print an error message if validation fails
            print(f"[✘] Error validating TIFF file {tiff_path}: {e}")
            return False  # Return False if the file is invalid

    def validate_zarr_with_spatialdata(self, zarr_path: str) -> bool:
        """
        Validate a .zarr directory using spatialdata.

        Args:
            zarr_path (str): Path to the .zarr directory.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Import spatialdata for validating .zarr files
            import spatialdata as sd
            # Attempt to read the .zarr file to validate its structure
            sd.read_zarr(zarr_path)
            return True  # Return True if validation succeeds
        except Exception as e:
            # Print an error message if validation fails
            print(f"[✘] SpatialData validation failed for {zarr_path}: {e}")
            return False  # Return False if validation fails

    def validate_h5ad_with_squidpy(self, h5ad_path: str) -> bool:
        """
        Validate an .h5ad file using squidpy's spatial omics utilities.

        Args:
            h5ad_path (str): Path to the .h5ad file.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Import squidpy and anndata for h5ad validation
            import squidpy as sq
            import anndata
            # Read the .h5ad file using anndata
            adata = anndata.read_h5ad(h5ad_path)
            # Perform a spatial neighbors check using squidpy
            sq.gr.spatial_neighbors(adata)
            return True  # Return True if validation succeeds
        except Exception as e:
            # Print an error message if validation fails
            print(f"[✘] Squidpy validation failed for {h5ad_path}: {e}")
            return False  # Return False if validation fails

    def validate_csv_with_scanpy(self, csv_path: str) -> bool:
        """
        Validate a .csv file for compatibility with single-cell data standards using scanpy.

        Args:
            csv_path (str): Path to the .csv file.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Import scanpy for additional single-cell data handling
            import scanpy as sc
            # Load the CSV file into a DataFrame
            data = pd.read_csv(csv_path)
            # Check if the expected columns exist in the DataFrame
            if "Gene" in data.columns and "Expression" in data.columns:
                return True  # Return True if the expected columns are present
            else:
                # Print an error if the columns are missing
                print(f"[✘] Scanpy validation failed: Missing expected columns in {csv_path}.")
                return False
        except Exception as e:
            # Print an error message if validation fails
            print(f"[✘] Scanpy validation failed for {csv_path}: {e}")
            return False  # Return False if validation fails

    def validate_required_keys(self, sdata):
        """
        Validate that the .zarr dataset contains all required keys.

        Args:
            sdata: SpatialData object to validate.

        Raises:
            KeyError: If any required key is missing.
        """
        # List of keys that must exist in the dataset
        required_keys = [
            'HE_original', 'HE_registered', 'HE_nuc_original', 'HE_nuc_registered',
            'DAPI', 'DAPI_nuc', 'anucleus', 'transcripts', 'group', 'group_HEspace'
        ]

        # Check if any required key is missing in the SpatialData object
        missing_keys = [key for key in required_keys if key not in sdata]

        # Raise an error if there are any missing keys
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")

    def validate_alignment(self, sdata):
        """
        Validate alignment between registered H&E images, segmentation masks, and transcriptomics data.

        Args:
            sdata: SpatialData object.

        Raises:
            ValueError: If any alignment issue is detected.
        """
        try:
            # Extract registered H&E image data as a NumPy array
            he_registered = sdata['HE_registered'].to_numpy()

            # Extract registered nucleus segmentation mask as a NumPy array
            he_nuc_registered = sdata['HE_nuc_registered'].to_numpy()

            # Extract DAPI nucleus segmentation mask as a NumPy array
            dapi_nuc = sdata['DAPI_nuc'].to_numpy()

            # Check if the shapes of the registered H&E image and nucleus mask match
            if he_registered.shape[:2] != he_nuc_registered.shape:
                raise ValueError("Mismatch: HE_registered vs HE_nuc_registered")

            # Check if the shapes of the registered H&E image and DAPI nucleus mask match
            if he_registered.shape[:2] != dapi_nuc.shape:
                raise ValueError("Mismatch: HE_registered vs DAPI_nuc")

            # Print a success message if all alignments are correct
            print("Alignment validation successful.")
        except KeyError as e:
            # Print an error if any required alignment key is missing
            print(f"Missing key for alignment validation: {e}")

    def validate_gene_expression(self, sdata):
        """
        Validate gene expression data in `anucleus`.

        Args:
            sdata: SpatialData object.

        Raises:
            ValueError: If gene expression validation fails.
        """
        try:
            # Access the `anucleus` dataset containing gene expression data
            anucleus = sdata['anucleus']

            # Verify that the gene count matches the expected value (460 genes)
            gene_count = anucleus.var.shape[0]
            if gene_count != 460:
                raise ValueError(f"Gene count mismatch: Expected 460, found {gene_count}")

            # Verify that the gene expression data is correctly log1p-normalized
            raw_counts = anucleus.layers['counts']
            normalized = anucleus.X
            if not np.allclose(np.log1p(raw_counts / raw_counts.sum(axis=1, keepdims=True) * 100), normalized):
                raise ValueError("Gene expression data is not correctly log1p-normalized")

            # Print a success message if gene expression validation passes
            print("Gene expression validation successful.")
        except KeyError as e:
            # Print an error if any required key for gene expression is missing
            print(f"Missing key for gene expression validation: {e}")

    def validate_splits(self, sdata):
        """
        Validate train/validation/test splits.

        Args:
            sdata: SpatialData object.

        Raises:
            ValueError: If split validation fails.
        """
        try:
            # Access the `group` column that defines train/validation/test regions
            group_values = sdata['group'].to_numpy()

            # Ensure that all group values are valid (0=train, 1=validation, 2=test, 4=no transcript-train)
            unique_groups = np.unique(group_values)
            if not all(val in [0, 1, 2, 4] for val in unique_groups):
                raise ValueError(f"Unexpected values in `group`: {unique_groups}")

            # Print a success message if the split validation passes
            print("Splits validation successful.")
        except KeyError as e:
            # Print an error if the `group` key is missing
            print(f"Missing key for split validation: {e}")

    def validate_spatial_data(self) -> Dict[str, Dict[str, str]]:
        """
        Validate spatial data files listed in the manifest.

        Returns:
            Dict[str, Dict[str, str]]: Validation results for each file in the manifest.
        """
        results = {}  # Dictionary to store validation results
        # Iterate over each Crunch and its files in the manifest
        for crunch_name, files in self.manifest.items():
            print(f"Validating spatial data for {crunch_name}...")
            results[crunch_name] = {}
            for file_path, file_info in files.items():
                full_path = os.path.join(self.base_dir, file_path)
                # Determine the file type and validate accordingly
                if file_path.endswith(".zarr"):
                    is_valid = self.validate_zarr_with_spatialdata(full_path)
                elif file_path.endswith(".tiff"):
                    is_valid = self.validate_tiff(full_path)
                elif file_path.endswith(".csv"):
                    is_valid = self.validate_csv_with_scanpy(full_path)
                elif file_path.endswith(".h5ad"):
                    is_valid = self.validate_h5ad_with_squidpy(full_path)
                else:
                    is_valid = False  # Mark as invalid if the file type is unknown
                # Store the validation result
                results[crunch_name][file_path] = "valid" if is_valid else "invalid"
        return results

    def run_validation(self):
        """
        Run all validation steps for the datasets listed in the manifest and save results.
        """
        # Initialize a dictionary to store validation results for each Crunch dataset
        results = {}

        # Iterate over each Crunch dataset and its files listed in the manifest
        for crunch_name, files in self.manifest.items():
            print(f"Validating {crunch_name}...")  # Print the name of the current Crunch being validated
            results[crunch_name] = {}  # Initialize results for the current Crunch

            # Iterate over each file associated with the current Crunch dataset
            for file_path, file_info in files.items():
                # Construct the full path to the file based on the base directory
                full_path = os.path.join(self.base_dir, file_path)

                # Check if the file is a .zarr file
                if file_path.endswith(".zarr"):
                    # Load the .zarr dataset using spatialdata
                    sdata = sd.read_zarr(full_path)
                    try:
                        # Validate that the dataset contains all required keys
                        self.validate_required_keys(sdata)

                        # Validate alignment between registered H&E images, segmentation masks, and transcriptomics data
                        self.validate_alignment(sdata)

                        # Validate the gene expression data for normalization and expected gene count
                        self.validate_gene_expression(sdata)

                        # Validate the proper assignment of train/validation/test splits
                        self.validate_splits(sdata)

                        # Validate nuclei ID consistency across different files in the dataset
                        self.validate_nuclei_ids(sdata)

                        # If all validations pass, mark the file as valid
                        results[crunch_name][file_path] = "valid"
                    except Exception as e:
                        # If any validation step fails, catch the exception and mark the file as invalid
                        print(f"Validation failed for {file_path}: {e}")
                        results[crunch_name][file_path] = "invalid"
                else:
                    # If the file type is unsupported, mark it as unsupported
                    results[crunch_name][file_path] = "unsupported"

        # Log the validation results to a file for later inspection
        self.log_validation_results(results)

        # Save the updated validation statuses back to the manifest
        self._save_manifest()

        # Print a message indicating that the validation process is complete
        print("Validation complete.")

    @staticmethod
    def print_results(results: Dict[str, Dict[str, str]]):
        """
        Print validation results in a readable format.

        Args:
            results (Dict[str, Dict[str, str]]): Validation results to print.
        """
        print("\nValidation Results:")
        # Iterate over the results and print them
        for crunch_name, files in results.items():
            print(f"\n{crunch_name}:")
            for file_path, status in files.items():
                print(f"  {file_path}: {status}")


# Main script execution
def main():
    """
    Main function for validating spatial data files.
    """
    # Define the base directory containing the data
    base_dir = "/mnt/d/AutoImmuneML"
    # Create an instance of the validator
    validator = SpatialDataValidator(base_dir)
    # Run the validation process
    validator.run_validation()


if __name__ == "__main__":
    main()
