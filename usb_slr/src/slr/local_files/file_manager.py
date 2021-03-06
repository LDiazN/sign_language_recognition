"""
    Wrapper class to manage local files
"""
# Python imports
from pathlib import Path
import zipfile

# Local imports 
from slr.config import settings

class FileManager:
    """
        Wrapper class to manage local files.
        # Parameters:
            - home_dir : `str` = path to home directory for the library 
    """

    def __init__(self, home_dir : str = settings.slr_home) -> None:
        # Set up home directory
        self._home_dir = Path(home_dir)

    @property
    def ms_dataset_dir(self) -> str:
        """
            Returns path to the microsoft dataset directory
        """
        return str(Path(self._home_dir, "ms_dataset"))
    
    @property
    def ms_dataset_description_dir(self) -> str:
        """
            Returns the path to the microsoft dataset description dir, usually inside the
            ms_dataset_dir path
        """
        return str(Path(self.ms_dataset_dir, "MS-ASL"))

    def store_ms_dataset(self, path : str):
        """
            Store in local files folder the microsoft dataset files. More info 
            about such dataset here: https://www.microsoft.com/en-us/download/details.aspx?id=100121

            # Parameters:
                - path: `str` = path to microsoft dataset in zip format
        """

        # Consistency check
        zip_path = Path(path)
        if not zip_path.exists():
            raise ValueError(f"Provided path to microsoft zip dataset is not a valid path or it does not exists. Path: {path}")

        # Unzip files
        with zipfile.ZipFile(path, 'r') as zip_ref:
            ms_dir_path = Path(self.ms_dataset_dir)
            if not ms_dir_path.exists():
                ms_dir_path.mkdir(parents=True)
            zip_ref.extractall(self.ms_dataset_dir)

