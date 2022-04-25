"""
    Load data from disk and prepare it to be consumed by the model
"""
# Third party imports 
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# Local imports 
from slr.dataset_manager.dataset_managers import DatasetManager
from slr.local_files.file_manager import FileManager
from slr.model.labels import Labels

# python imports
import numpy as np
from typing import Tuple

class DataIngestor:
    """
        Load data from disk to create a dataset that you can feed to the model
    """
    
    # Maximum ammount of rows for a sign. Signs with a shorter length will be padded with 0, 
    # with higher length will be truncated
    MAX_SIGN_LEN = 180

    def __init__(self, file_manager : FileManager = FileManager()):
        self._labels = Labels(file_manager)
        self._file_manager = file_manager
        self._dataset_manager = DatasetManager(file_manager)

    @property
    def labels(self) -> Labels:
        return self._labels

    def generate_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Create the data to be feeded into the model

            # Return 
            (Training dataset, test dataset, training labels, test labels)
        """
        # Retrieve data 
        videos, labels = [], []
        for features, description in self._dataset_manager.test_numeric_dataset_client.retrieve_data(): # TODO use training data as well, not only test data
            
            # Get the feature shape
            rows, _ = features.shape

            if rows < DataIngestor.MAX_SIGN_LEN:
                # Make it bigger
                features = np.pad(features, [(0, DataIngestor.MAX_SIGN_LEN - rows), (0,0)])
            elif rows > DataIngestor.MAX_SIGN_LEN:
                # Make it smaller
                features = features[:DataIngestor.MAX_SIGN_LEN]


            videos.append(features)
            labels.append(description.label)

        # Create matrices
        X = np.array(videos)

        # assign labels
        Y = to_categorical(labels).astype(int)

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)
        return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
