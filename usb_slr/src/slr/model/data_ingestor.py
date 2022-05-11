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
from typing import Tuple, Optional, Callable
import logging

class DataIngestor:
    """
        Load data from disk to create a dataset that you can feed to the model
    """
    
    # Maximum ammount of rows for a sign. Signs with a shorter length will be given a configurable padding
    # that defaults to 0, with higher length will be truncated
    MAX_SIGN_LEN = 80

    def __init__(self, file_manager : FileManager = FileManager()):
        self._labels = Labels(file_manager)
        self._file_manager = file_manager
        self._dataset_manager = DatasetManager(file_manager)

    @property
    def labels(self) -> Labels:
        return self._labels

    def generate_train_data(self, frame_limit : Optional[int] = None, video_limit : Optional[int] = None, padding_func : Optional[ Callable[[np.ndarray],np.ndarray] ] = None ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Create the data to be feeded into the model

            Some variables for generation are configurable:
            * frame_limit : limit of frames per video
            * video_limit : limit of videos for training data
            * padding_func : padding function to use

            # Return 
            (Training dataset, test dataset, training labels, test labels)
        """


        # Configurable limit of frames used per video
        FRAME_CAP = DataIngestor.MAX_SIGN_LEN if frame_limit is None else frame_limit
        video_data = self._dataset_manager.train_numeric_dataset_client.retrieve_data()

        # Retrieve data 
        videos, labels = [], []
        for idx, (features, description) in enumerate( video_data ): # TODO use training data as well, not only test data
            logging.warning("Generating data for video %d", idx)
            
            # Get the feature shape
            rows, _ = features.shape

            if rows < FRAME_CAP:
                # Configurable padding function
                padder = self.pad_with_zeroes if padding_func is None else padding_func

                features = padder(features)
            elif rows > FRAME_CAP: 
                # Make it smaller
                features = features[:FRAME_CAP]

            videos.append(features)
            labels.append(description.label)

            # Configurable video limit
            if video_limit is not None and idx == video_limit:
                break

        # Create matrices
        X = np.array(videos)

        # assign labels
        Y = to_categorical(labels).astype(int)

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)
        return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

    def pad_with_repetitions(self, frames: np.ndarray) -> np.ndarray :
        """ Pad feature list with repetitions on both extremes """
        diff = DataIngestor.MAX_SIGN_LEN - len(frames)

        # Frames to repeat
        first, last = frames[0], frames[-1:][0]
        # Number of repetitions
        low, high = int( np.floor(diff/2) ), int( np.ceil(diff/2) )

        # pad begginning with first frames
        for _ in range(low):
            frames = np.concatenate( ([first], frames) )
            
        # pad ending with last frames
        for _ in range(high):
            frames = np.concatenate( (frames, [last]) ) 

        return frames 

    def pad_with_zeroes(self, frames: np.ndarray) -> np.ndarray:
        rows, _ = frames.shape
        return  np.pad(frames, [(0, DataIngestor.MAX_SIGN_LEN - rows), (0,0)])
