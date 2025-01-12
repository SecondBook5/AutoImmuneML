# File: src/validators/spatial_alignment_validator.py

import numpy as np
from skimage.measure import regionprops
from typing import Dict, Any

class SpatialAlignmentValidator:
    """
    Validator for ensuring the spatial alignment and integrity of spatial transcriptomics data.

    This validator ensures:
    - Matching of nucleus segmentation masks with cell IDs in gene expression data.
    - Correct coordinate alignment between segmentation masks and registered images.
    - Consistency of spatial data attributes across input datasets.
    """

    def validate(self, sdata: Dict[str, Any]) -> bool:
        """
        Validate the spatial alignment and integrity of the input data.

        Args:
            sdata (Dict[str, Any]): SpatialData-like dictionary containing the input data components.
                Expected keys include:
                - 'HE_nuc_original': Nucleus segmentation mask (original image coordinate system).
                - 'HE_nuc_registered': Nucleus segmentation mask (registered image coordinate system).
                - 'HE_registered': Registered H&E image.
                - 'anucleus': Gene expression data with cell IDs.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        # Validate required keys
        required_keys = ['HE_nuc_original', 'HE_nuc_registered', 'HE_registered', 'anucleus']
        for key in required_keys:
            if key not in sdata:
                raise ValueError(f"Missing required key in SpatialData: {key}")

        # Validate matching dimensions
        self._validate_dimensions(
            sdata['HE_nuc_registered'].shape, sdata['HE_registered'].shape, 'HE_nuc_registered', 'HE_registered'
        )

        # Validate cell IDs in segmentation mask match those in gene expression data
        self._validate_cell_ids(sdata['HE_nuc_original'], sdata['anucleus'])

        # Validate spatial coordinate alignment (optional, depends on use case)
        if 'transcripts' in sdata:
            self._validate_transcript_coordinates(sdata['transcripts'], sdata['HE_registered'].shape)

        print("[INFO] Spatial alignment validation passed.")
        return True

    def _validate_dimensions(self, mask_shape: tuple, image_shape: tuple, mask_name: str, image_name: str):
        """
        Validate that the dimensions of the segmentation mask and image match.

        Args:
            mask_shape (tuple): Shape of the segmentation mask.
            image_shape (tuple): Shape of the image.
            mask_name (str): Name of the mask (for error messages).
            image_name (str): Name of the image (for error messages).

        Raises:
            ValueError: If dimensions do not match.
        """
        if mask_shape[1:] != image_shape[1:]:
            raise ValueError(f"Dimension mismatch: {mask_name} ({mask_shape}) and {image_name} ({image_shape})")

    def _validate_cell_ids(self, mask, anucleus):
        """
        Validate that cell IDs in the segmentation mask match those in the gene expression data.

        Args:
            mask (np.ndarray): Nucleus segmentation mask.
            anucleus (Any): Gene expression data (expected to have a cell_id column).

        Raises:
            ValueError: If there are mismatched or missing cell IDs.
        """
        mask_cell_ids = np.unique(mask)
        mask_cell_ids = mask_cell_ids[mask_cell_ids > 0]  # Exclude background (ID = 0)
        anucleus_cell_ids = anucleus.obs['cell_id'].values

        missing_ids = set(anucleus_cell_ids) - set(mask_cell_ids)
        extra_ids = set(mask_cell_ids) - set(anucleus_cell_ids)

        if missing_ids:
            raise ValueError(f"Missing cell IDs in segmentation mask: {missing_ids}")
        if extra_ids:
            print(f"[WARNING] Extra cell IDs in segmentation mask: {extra_ids}")

    def _validate_transcript_coordinates(self, transcripts, image_shape):
        """
        Validate that transcript coordinates are within the bounds of the image.

        Args:
            transcripts (np.ndarray): Array of transcript coordinates (x, y).
            image_shape (tuple): Shape of the image to validate against.

        Raises:
            ValueError: If coordinates are out of bounds.
        """
        x_coords, y_coords = transcripts[:, 0], transcripts[:, 1]
        if np.any(x_coords < 0) or np.any(x_coords >= image_shape[2]):
            raise ValueError("Transcript x-coordinates are out of bounds.")
        if np.any(y_coords < 0) or np.any(y_coords >= image_shape[1]):
            raise ValueError("Transcript y-coordinates are out of bounds.")
