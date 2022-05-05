"""
    Load data from disk and prepare it to be consumed by the model
"""
# Third party imports 
import random

from pandas import array
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# Local imports 
from slr.dataset_manager.dataset_managers import DatasetManager, SignDescription
from slr.local_files.file_manager import FileManager
from slr.model.labels import Labels

# python imports
import numpy as np
from typing import Callable, Optional, Tuple

class DataIngestor:
    """
        Load data from disk to create a dataset that you can feed to the model
    """
    
    # Maximum ammount of rows for a sign. Signs with a shorter length will be padded with 0, 
    # with higher length will be truncated
    MAX_SIGN_LEN = 60

    def __init__(self, file_manager : FileManager = FileManager()):
        self._labels = Labels(file_manager)
        self._file_manager = file_manager
        self._dataset_manager = DatasetManager(file_manager)

    @property
    def labels(self) -> Labels:
        return self._labels

    def generate_train_data(self, pad_with_random : bool = False, expand_if_small : bool = False, predicate : Optional[Callable[[np.ndarray, SignDescription], bool]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Create the data to be feeded into the model

            # Return 
            (Training dataset, test dataset, training labels, test labels)
        """
        # Retrieve data 
        videos, labels = [], []
        for features, description in self._dataset_manager.test_numeric_dataset_client.retrieve_data(): # TODO use training data as well, not only test data
            
            # Skip this row if does not matches predicate
            if predicate and not predicate(features, description):
                continue

            # Each row is a single frame
            rows, _ = features.shape
            if expand_if_small and rows < DataIngestor.MAX_SIGN_LEN:
                features = self._expand_dataset_to_size(features, DataIngestor.MAX_SIGN_LEN, pad_with_random)
            else:
                features = self._pad(features, DataIngestor.MAX_SIGN_LEN, pad_with_random)

            videos.append(features)
            labels.append(description.label)

        # Create matrices
        X = np.array(videos)

        # assign labels
        Y = to_categorical(labels).astype(int)

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)
        return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

    def _pad(self, array : np.ndarray, valid_rows : int, fill_random : bool = False) -> np.ndarray:
        """
            Pad an array, returning either an array that's smaller than the original, or 
            an array with random filling depending on if the array size is smaller or bigger than valid_rows
        """

        rows, cols = array.shape

        if rows < valid_rows:
            # Make it bigger
            array = np.pad(array, [(0, valid_rows - rows), (0,0)])
        elif rows > DataIngestor.MAX_SIGN_LEN:
            # Make it smaller
            array = array[:DataIngestor.MAX_SIGN_LEN]

        # Fill padding with random values if requested so
        if fill_random:
            for i in range(0, valid_rows):
                for j in range(0, cols):
                    array[i,j] = random.random() * 10 - 3

        return array

    def _expand_dataset_to_size(self, matrix : np.ndarray, target_size : int, fill_random : bool = False) -> np.ndarray:
        """
            Use this function to expand a list of rows 
        """
        rows, cols = matrix.shape
        assert target_size > rows


        # Create resulting array
        result = np.zeros((target_size, cols))

        # How many rows we have to fill
        to_fill = target_size - rows

        # Count how many spaces between 
        pad_size = to_fill//(rows - 1)

        curr_row = 0
        for i in range(0, rows - 1):
            prev_row = matrix[i]
            next_row = matrix[i+1]

            result[curr_row] = prev_row; curr_row += 1
            for j in range(1, pad_size+1):

                alpha = j/(pad_size + 1)
                result[curr_row] = (prev_row + alpha*(next_row - prev_row))
                curr_row += 1

        # Add last row
        result[curr_row] = matrix[rows-1]
        curr_row += 1

        # padding with random if requested so
        if fill_random:
            for i in range(curr_row, target_size):
                for j in range(cols):
                    result[i,j] = random.random() * 6 - 3

        
        return result