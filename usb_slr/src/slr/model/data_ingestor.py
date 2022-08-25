"""
    Load data from disk and prepare it to be consumed by the model
"""
# Third party imports 
import random
from tkinter import Frame

from pandas import array
from slr.data_processing.image_parser import PoseValues
import torch
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# Local imports 
from slr.dataset_manager.dataset_managers import MicrosoftDatasetManager, SignDescription
from slr.local_files.file_manager import FileManager
from slr.model.labels import Labels

# python imports
import numpy as np
from typing import Dict, Iterable, Tuple, Optional, Callable
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
        self._dataset_manager = MicrosoftDatasetManager(file_manager)

    def retrieve_all(self) -> Iterable[Tuple[np.ndarray, SignDescription]]:
        """Retrieve all information in the provided dataset

        Returns:
            Iterable[Tuple[np.ndarray, SignDescription]]: All information available in the dataset, train, test, or valid
        """

        for x in self._dataset_manager.train_numeric_dataset_client.retrieve_data():
            yield x
        
        for x in self._dataset_manager.val_numeric_dataset_client.retrieve_data():
            yield x

        for x in self._dataset_manager.test_numeric_dataset_client.retrieve_data():
            yield x

    @property
    def labels(self) -> Labels:
        return self._labels

    def generate_train_data(
            self, 
            predicate       : Optional[ Callable[ [np.ndarray, SignDescription], bool] ] = None, 
            frame_limit     : Optional[int] = None, 
            video_limit     : Optional[int] = None, 
            padding_func    : Optional[ Callable[ [np.ndarray],np.ndarray] ] = None ,
            normalize_location : bool = False,
            reduce_labels : bool = False,
            ignore_face : bool = False,
            samples_limit : Optional[int] = None,
            test_size : float = 0.2

        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[int, str]]]:
        """
            Create the data to be feeded into the model

            Some variables for generation are configurable:
            * frame_limit : limit of frames per video
            * video_limit : limit of videos for training data
            * padding_func : padding function to use
            * normalize_location : `bool` = if this data should be location-normalized
            * reduce_labels : `bool` = if should reduce the labels vector to the ones just in the resulting dataset. 
                For example, if the dataset has 1000 labels, and the resulting dataset uses only 10, 
                the returning label vector will have only 10 columns

            # Return
            (Training dataset, test dataset, training labels, test labels, label map | None)
            Where labelmap is a dict mapping from reduced labels to actual labels, if reduce_labels == True
        """

        # Configurable limit of frames used per video
        FRAME_CAP  = DataIngestor.MAX_SIGN_LEN if frame_limit is None else frame_limit
        video_data = self.retrieve_all()

        # Retrieve data 
        videos, labels = [], []

        for idx, (features, description) in enumerate( video_data ): 
            
            # Skip this row if does not matches predicate
            if predicate and not predicate(features, description):
                continue
            if features.shape == (0,):
                print(f"[Warning] There's a sign description with no signs: {description.file}")
                continue

            # Check if enough samples are already taken
            if samples_limit and idx >= samples_limit:
                break

            logging.warning("Generating data for video %d", idx)

            # Get the feature shape
            rows, _ = features.shape

            # normalize location if requested so
            if normalize_location:
                for (i,row) in enumerate(features):
                    pose_value = PoseValues.from_array(row)
                    pose_value.normalize_location()
                    features[i] = pose_value.concatenated(exclude_face=ignore_face)

            if rows < FRAME_CAP:
                # Configurable padding function
                transf   =  FeatureTransformer(FRAME_CAP)
                features = transf.pad_with_zeroes(features) if padding_func is None else padding_func(transf, features) # tpye: ignore

            elif rows > FRAME_CAP: 
                # Make it smaller
                sample = np.random.choice([i for i in range(len(features))], FRAME_CAP, replace=False)
                sample.sort()
                features = features[sample]

            videos.append(features)
            labels.append(description.label)
            # Configurable video limit
            if video_limit is not None and idx == video_limit:
                break

        # Create matrices
        X = np.array(videos)

        # assign labels
        Y = to_categorical(labels).astype(int)

        if reduce_labels:
            Y, labelmap = self._reduce_labels(Y)
        else:
            labelmap = None

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=test_size)
        return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, labelmap

    def _reduce_labels(self, labels : np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        actual_labels = labels.argmax(1)
        labelset = {l for l in actual_labels}
        labelmap = { l: i for (i, l) in enumerate(labelset)}

        new_labels = np.zeros((labels.shape[0], len(labelset)))

        for (i, l) in enumerate(actual_labels):
            new_labels[i, labelmap[l]] = 1

        i_to_str_labelmap = self._dataset_manager.label_map
        # Return labelmap to recover information about what label is what when we reduce labels 
        labelmap = {i : i_to_str_labelmap[l] for (l, i) in labelmap.items() }


        return new_labels, labelmap

# TODO, include simmetric padding, and a better way to set padding up
class FeatureTransformer:
    """ Convenient class to manage different types of padding over bidimentional feature arrays"""

    def __init__(self, limit : int) : 
        self.LIMIT  = limit

    @property
    def limit(self):
        return self.LIMIT

    @limit.setter
    def limit(self, val : int):
        self.LIMIT = val

    def pad_with_repetitions(self, features : np.ndarray ) -> np.ndarray :
        """ Pad feature list with repetitions on both extremes """

        diff = self.LIMIT - len(features)

        # features to repeat
        first, last = features[0], features[-1:][0]

        # Number of repetitions
        low, high = int( np.floor(diff/2) ), int( np.ceil(diff/2) )

        # pad begginning with first features
        for _ in range(low):
            features = np.concatenate( ([first], features) )
            
        # pad ending with last features
        for _ in range(high):
            features = np.concatenate( (features, [last]) ) 

        return features 

    def pad_with_zeroes(self, features : np.ndarray ) -> np.ndarray:
        """ Add trailing zeroes padding to feature list """
        rows, _ = features.shape
        diff = self.LIMIT - rows
        features = np.pad(features, [(0, diff), (0,0)])

        return features

    def pad_with_random(self, features : np.ndarray )  -> np.ndarray:
        """ Replace input feature matrix with mock random data. """ 

        rows, cols = features.shape
        features = np.resize( features, (self.LIMIT, cols) ) 

        for i in range(rows, self.LIMIT):
            for j in range(0, cols):
                features[i,j] = random.random() * 10 - 3

        return features

    def expand_dataset_to_size(self, matrix : np.ndarray, fill_random : bool = False) -> np.ndarray:
        """
            Use this function to expand a list of rows 
        """
        rows, cols = matrix.shape
        assert self.LIMIT > rows


        # Create resulting array
        result = np.zeros((self.LIMIT, cols))

        # How many rows we have to fill
        to_fill = self.LIMIT - rows

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
            for i in range(curr_row, self.LIMIT):
                for j in range(cols):
                    result[i,j] = random.random() * 6 - 3

        
        return result