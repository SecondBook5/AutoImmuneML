import tifffile as tiff
from src.loaders.base_loader import BaseLoader

class TIFFLoader(BaseLoader):
    """
    Loader for .tiff files, inheriting from BaseLoader.
    """

    def load(self) -> any:
        """
        Load the .tiff file and return the image data as a numpy array.

        Returns:
            numpy.ndarray: The loaded TIFF image data.
        """
        return tiff.imread(self.path)
