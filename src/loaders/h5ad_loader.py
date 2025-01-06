import anndata as ad
from src.loaders.base_loader import BaseLoader

class H5ADLoader(BaseLoader):
    """
    Loader for .h5ad files, inheriting from BaseLoader.
    """

    def load(self) -> ad.AnnData:
        """
        Load the .h5ad file and return the AnnData object.

        Returns:
            anndata.AnnData: The loaded AnnData object.
        """
        return ad.read_h5ad(self.path)