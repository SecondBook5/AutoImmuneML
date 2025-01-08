# File: src/preprocessors/spatial_preprocessor.py
import os
import numpy as np
import tifffile
import h5py
from skimage.measure import regionprops
from tqdm import tqdm
from typing import Dict, Any
from src.preprocessors.base_preprocessor import BasePreprocessor


class ImagePreprocessor(BasePreprocessor):
    """
    Preprocessor for H&E images to prepare input data for downstream tasks.

    Features:
    - Validation of input SpatialData objects.
    - Extraction of image patches centered on nuclei.
    - Normalization of pixel intensities.
    - Saving patches and metadata in HDF5 format.
    """

    def preprocess(self, sdata, **kwargs) -> Dict[str, Any]:
        """
        Main preprocessing pipeline for H&E images.

        Args:
            sdata: SpatialData object containing image and nucleus data.
            kwargs: Additional parameters, including:
                - crop_size: Size of cropped patches (default: 128).
                - output_path: File path to save processed outputs.

        Returns:
            Dict[str, Any]: Metadata of processed image patches.
        """
        # Get the crop size from kwargs or set to default of 128
        crop_size = kwargs.get("crop_size", 128)

        # Get the output path for saving patches from kwargs
        output_path = kwargs.get("output_path", "processed_patches.h5")

        # Validate the input SpatialData object
        print("Validating input SpatialData object...")
        self.validate(sdata)

        # Extract regions (nuclei) from the nucleus segmentation image
        print("Extracting nucleus regions...")
        regions = regionprops(sdata['HE_nuc_registered'][0, :, :].to_numpy())

        # Get the registered H&E image as a NumPy array
        print("Loading registered H&E intensity image...")
        intensity_image = sdata['HE_registered'].to_numpy()

        # Extract patches and save them in HDF5 format
        print("Extracting and saving patches...")
        metadata = self._extract_and_save_patches(
            regions, intensity_image, crop_size, output_path
        )

        # Return metadata containing patch information
        return metadata

    def validate(self, sdata):
        """
        Validate the integrity of the input SpatialData object.

        Args:
            sdata: SpatialData object to validate.

        Raises:
            ValueError: If required attributes are missing or invalid.
        """
        # Ensure the SpatialData object contains the required keys
        if 'HE_nuc_registered' not in sdata or 'HE_registered' not in sdata:
            raise ValueError("SpatialData object must contain 'HE_nuc_registered' and 'HE_registered' attributes.")

        # Ensure the nucleus segmentation and H&E image dimensions match
        if sdata['HE_nuc_registered'].shape[1:] != sdata['HE_registered'].shape[1:]:
            raise ValueError("Dimension mismatch between 'HE_nuc_registered' and 'HE_registered'.")

    def _extract_and_save_patches(self, regions, intensity_image, crop_size, output_path):
        """
        Extract patches and save them in an HDF5 file.

        Args:
            regions: List of nucleus regions from regionprops.
            intensity_image: Registered H&E image as a NumPy array.
            crop_size: Size of the square crop.
            output_path: Path to save the HDF5 file.

        Returns:
            Dict[str, Any]: Metadata of the extracted patches.
        """
        # Calculate half of the crop size for boundary adjustments
        half_crop = crop_size // 2

        # Initialize lists to store patch data and metadata
        patches = []
        metadata = []

        # Open an HDF5 file for writing patch data
        with h5py.File(output_path, 'w') as h5f:
            # Create datasets for patches and metadata
            h5f.create_dataset("patches", shape=(0, crop_size, crop_size, intensity_image.shape[0]),
                               maxshape=(None, crop_size, crop_size, intensity_image.shape[0]), dtype="float32",
                               chunks=True)
            h5f.create_dataset("metadata", shape=(0, 3), maxshape=(None, 3), dtype="int")

            # Loop through each nucleus region
            for props in tqdm(regions, desc="Processing nuclei"):
                # Get the unique label ID and centroid coordinates of the nucleus
                cell_id = props.label
                centroid = props.centroid
                y_center, x_center = int(centroid[0]), int(centroid[1])

                # Calculate crop boundaries
                minr, maxr = y_center - half_crop, y_center + half_crop
                minc, maxc = x_center - half_crop, x_center + half_crop

                # Adjust boundaries to stay within image dimensions
                minr, maxr = max(0, minr), min(intensity_image.shape[1], maxr)
                minc, maxc = max(0, minc), min(intensity_image.shape[2], maxc)

                # Extract the crop and pad if necessary
                crop = intensity_image[:, minr:maxr, minc:maxc]
                if crop.shape[1] != crop_size or crop.shape[2] != crop_size:
                    crop = np.pad(crop, ((0, 0), (0, crop_size - crop.shape[1]), (0, crop_size - crop.shape[2])),
                                  mode="constant", constant_values=0)

                # Normalize the patch pixel intensities to [0, 1]
                crop = crop.astype("float32") / 255.0

                # Append the crop and metadata to respective lists
                patches.append(crop)
                metadata.append((cell_id, y_center, x_center))

                # Save the patch and metadata to the HDF5 file
                h5f["patches"].resize(h5f["patches"].shape[0] + 1, axis=0)
                h5f["patches"][-1] = crop
                h5f["metadata"].resize(h5f["metadata"].shape[0] + 1, axis=0)
                h5f["metadata"][-1] = [cell_id, y_center, x_center]

        # Return metadata as a dictionary
        return {"patches": len(patches), "output_path": output_path}
