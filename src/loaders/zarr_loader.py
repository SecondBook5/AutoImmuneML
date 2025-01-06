import zarr
from src.loaders.base_loader import BaseLoader

class ZARRLoader(BaseLoader):
    """
    Loader for .zarr files, inheriting from BaseLoader.
    """

    def load(self) -> zarr.Group:
        """
        Load the .zarr file and return the Zarr group.

        Returns:
            zarr.Group: The loaded Zarr group.
        """
        return zarr.open_group(self.path, mode='r')
