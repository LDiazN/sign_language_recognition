"""
    Manage labels for this dataset. Labels come from the microsoft dataset description
"""

# Python imports
from typing import Dict
import numpy as np

# Local imports
from slr.local_files.file_manager import FileManager
from slr.dataset_manager.dataset_managers import DatasetManager


class Labels:
    """
        Model labels
    """

    def __init__(self, file_manager : FileManager):
        self._file_manager = file_manager
        self._dataset_manager = DatasetManager(file_manager)
        
    @property
    def file_manager(self) -> FileManager:
        """
            Object to manage local files and datasets
        """
        return self._file_manager

    @property
    def dataset_manager(self) -> DatasetManager:
        """
            Object to manage datasets locations and values
        """
        return self._dataset_manager

    @property
    def label_map(self) -> Dict[int, str]:
        """
            Object mapping from words to ints
        """
        return self.dataset_manager.label_map
    
    @property
    def label_map_inverse(self) -> Dict[str, int]:
        """
            Return the reverse map
        """
        return {y : x for (x, y) in self.label_map.items()}

    @property
    def labels_array(self) -> np.ndarray:
        """
            An array with all labels
        """
        return np.array([k for k in self.label_map.values()])