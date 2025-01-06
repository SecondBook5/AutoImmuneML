# File: src/loaders/csv_loader.py
import pandas as pd
from src.loaders.base_loader import BaseLoader

class CSVLoader(BaseLoader):
    """
    Loader for .csv files, inheriting from BaseLoader.
    """

    def load(self) -> pd.DataFrame:
        """
        Load the .csv file and return it as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The loaded CSV data.
        """
        return pd.read_csv(self.path)
