"""
    This file contains the necessary objects and operations 
    to manage datasets 
"""

# Third party imports
import logging
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
import termcolor

# Python imports
import dataclasses
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from typing_extensions import Self 
from pathlib import Path
import json
import pickle
import glob

# Local imports
from slr.local_files.file_manager import FileManager
from slr.dataset_manager.youtube_downloader import YoutubeDownloader
from slr.data_processing.video_converter import VideoConverter


@dataclasses.dataclass(eq=True, unsafe_hash=True)
class SignDescription:
    """
        Represents a sign description as depicted in the microsoft dataset
    """
    org_text    : str
    clean_text  : str
    start_time  : float
    signer_id   : int
    signer      : int
    start       : int
    end         : int
    file        : str
    label       : int
    height      : int
    width       : int
    fps         : float
    end_time    : float
    url         : str
    text        : str
    review      : Optional[int]
    box         : List[float] = dataclasses.field(hash=False)

    @property
    def dict(self):
        """
            return a dict representation
        """
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls : Type[Self], dict : Dict[str, Any]) -> Self:
        """
            Create a new description from a dict 
        """
        if "review" not in dict:
            dict["review"] = None
        return cls(**dict)


    @property
    def json(self) -> str:
        """
            Return a json string from this object
        """
        return json.dumps(self.dict)

class NumericDatasetClient:
    """
        Represents an interface to create and manage a numeric dataset.
        A numeric dataset is a numeric representation of the video dataset, 
        and is what you feed to the models. 

        For example, if you want to create a training dataset, provide this client
        with the folder where such dataset will be stored, and the folder where the corresponding 
        video files are stored.

        # Parameters
            - root_dir : `str` = Where the datasets will be stored and manager
            - video_datasets_dir : `str` = Where to find for videos to convert into datasets
    """

    def __init__(self, root_dir : str, video_datasets_dir : str):
        root_dir_path = Path(root_dir)

        # Check if files exists
        if not root_dir_path.exists():
            raise FileNotFoundError(f"Provided root file for NumericDatasetClient object does not exists: {root_dir}")
        
        video_datasets_path = Path(video_datasets_dir)

        # Check if files exists
        if not video_datasets_path.exists():
            raise FileNotFoundError(f"Provided root file for NumericDatasetClient object does not exists: {root_dir}")
        
        self._root_dir = root_dir
        self._video_datasets_dir = video_datasets_dir

    @property
    def _index_file_name(self) -> str:
        """
            Name of the index file used to recover dataset mapping from folder with files
        """
        return ".index.pkl"
    
    def load_index_file(self) -> Optional[Dict[str, Any]]:
        """
            Try to load the index file from the dataset path. 
            Index file has the following fields:
                - next_id : `int` = next valid id
                - mappings : `Dict[int, SignDescription]` = mappings from id to sign
        """
        path = Path(self.root_dir, self._index_file_name)

        if not path.exists():
            return None

        with path.open('rb') as f:
            index = pickle.load(f)

        return index
        
    def load_index_file_or_default(self) -> Dict[str, Any]:
        return self.load_index_file() or {'next_id' : 0, 'mappings' : {}}

    def save_index_file(self, index : Dict[str, Any]):
        """
            Try to save specified index file.
        """
        path = Path(self.root_dir, self._index_file_name)

        with path.open("wb") as f:
            pickle.dump(index, f)

    @property
    def root_dir(self) -> str:
        """
            Return root dir where this dataset is stored
        """
        return self._root_dir

    @property
    def video_datasets_dir(self) -> str:
        """
            Path where to look for videos datasets
        """
        return self._video_datasets_dir

    def _save_vid_fn(self, sign : SignDescription, features : np.ndarray):
        """
            Create a function to save the features for a single vid
        """
        index_file = self.load_index_file_or_default()
    
        next_id = index_file['next_id']

        # Save array as csv
        path = Path(self.root_dir, f"{next_id}.csv")
        self._save_vid_features(features, str(path))

        # Update index file
        index_file['mappings'][next_id] = sign
        index_file['next_id'] += 1

        self.save_index_file(index_file)

    def _save_vid_features(self, features : np.ndarray, name : str):
        """
            Save a single feature matrix as csv
        """
        np.savetxt(name, features, delimiter=",")

    def _load_vid_features(self, name : str) -> np.ndarray:
        """
            Load a single feature matrix from a csv
        """
        return np.loadtxt(name, delimiter=",")

    def save_sign_features_from_description(self, signs : List[SignDescription], model : Holistic, display_vids : bool = False, skip_if_in_index : bool = True):
        """
            Save features in disk
        """
        self.process_signs_from_description(signs, model, display_vids, self._save_vid_fn, skip_if_in_index)

    def process_signs_from_description(self, signs : List[SignDescription], model : Holistic, display_vids : bool = False, process_fn : Optional[Callable[[SignDescription, np.ndarray], None]] = None, skip_if_in_index : bool = True):
        """
            Create a dataframe from a list of videos
            # Parameters
                - signs : `[SignDescription]` = List of signs to parse
                - model : `Holistic` = Model to use to parse skeletal data from images
                - display_vids : `bool` = (optional) If should display videos as they're parsed
        """

        # Create video converter from holistic model
        video_converter = VideoConverter.from_holistic_model(model)

        for sign in signs:

            if skip_if_in_index:
                # Skip sign if already in index
                index_file = self.load_index_file_or_default()
                known_signs = set(index_file['mappings'].values())
                if sign in known_signs:
                    continue

            # Check if sign file exists
            file_path = self._get_vid_path_from_name(sign.file + ".mp4")
            if not file_path.exists():
                logging.warn(termcolor.colored(f"Could not parse video file {sign.file}, it does not exists. Skiping sign: {sign}", "yellow") )
                continue

            # If exists, extract features
            vid_features = self.create_feature_matrix_from_description(sign, video_converter, display_vids)

            # Process vid if requested
            if process_fn:
                process_fn(sign, vid_features)

    def _get_vid_path_from_name(self, vid_name : str) -> Path:
        """
            Create path for a video given its name
        """
        return Path(self._video_datasets_dir, vid_name)
        

    def create_feature_matrix_from_description(self, sign : SignDescription, video_converter : VideoConverter, display_video : bool = False) -> np.ndarray:
        """
            Create a feature matrix from a single, returning an array with the shape (number_of_frames, 1662), 
            1662 is the size of a frame data vector.
        """

        file_path = self._get_vid_path_from_name(sign.file + ".mp4")

        assert file_path.exists(), "Create row function assumes that the file does exists"

        data = video_converter.parse_video_from_file(str(file_path), sign.start_time, sign.end_time, display_video, sign.fps)
        as_arrays = [d.concatenated() for d in data]

        return np.array(as_arrays)

    def retrieve_data(self) -> Iterable[Tuple[np.ndarray, SignDescription]]:
        """
            Load data from this folder. The first returning list is a list of two-dimensional
            matrices, each corresponding to a single video. The second list is the sign description for 
            the sign underlying each matrix. The order of the first list matches the order of the second list
        """

        # Try to load index file
        index_file = self.load_index_file()

        # Raise error if no index file was found
        if index_file is None:
            raise FileNotFoundError(f"Could not finde index file for dataset in {self.root_dir}. Are you sure there's a valid dataset here?")

        # open every csv file from this folder 
        path_to_open = Path(self.root_dir, "*.csv")

        for file in glob.glob(str(path_to_open)):
            # Get index of this file, as it is the key to the index file mapping
            file_path = Path(file)
            file_num = int(file_path.name.strip(".csv"))

            # Get sign description
            yield self._load_vid_features(file), index_file['mappings'][file_num]

class DatasetManager:
    """
        Use this class to run common operations for datasets stored locally
    """

    def __init__(self, file_manager : FileManager = FileManager()):
        self._file_manager = file_manager
        

    @property
    def file_manager(self) -> FileManager:
        """
            File manager used by this object to manage datasets
        """
        return self._file_manager

    @property
    def train_dataset_name(self) -> str:
        """
            Name for the train dataset 
        """
        return "MSASL_train.json"

    @property
    def test_dataset_name(self) -> str:
        """
            Name for the test dataset 
        """
        return "MSASL_test.json"

    @property
    def val_dataset_name(self) -> str:
        """
            Name for the validation dataset 
        """
        return "MSASL_val.json"

    @property
    def classes_description_json_name(self) -> str:
        """
            Name of the file with classes description
        """
        return "MSASL_classes.json"

    @property
    def synonym_description_json_name(self) -> str:
        """
            Name of the file with synonim description
        """
        return "MSASL_synonym.json"

    @property
    def train_dataset_dir(self) -> str:
        """
            Where the actual train videos are stored
        """
        return str(Path(self.file_manager.ms_dataset_dir, "train_vids"))


    @property
    def test_dataset_dir(self) -> str:
        """
            Where the actual test videos are stored
        """
        return str(Path(self.file_manager.ms_dataset_dir, "test_vids"))

    @property
    def val_dataset_dir(self) -> str:
        """
            Where the actual validation videos are stored
        """
        return str(Path(self.file_manager.ms_dataset_dir, "val_vids"))

    @property
    def train_numeric_dataset_dir(self) -> str:
        """
            Path where to store the numeric train dataset generated from a video dataset using mediapipe
        """
        return str(Path(self.file_manager.ms_dataset_dir, "train_dataset_features"))

    @property
    def test_numeric_dataset_dir(self) -> str:
        """
            Path where to store the numeric test dataset generated from a video dataset using mediapipe
        """
        return str(Path(self.file_manager.ms_dataset_dir, "test_dataset_features"))

    @property
    def val_numeric_dataset_dir(self) -> str:
        """
            Path where to store the numeric validation dataset generated from a video dataset using mediapipe
        """
        return str(Path(self.file_manager.ms_dataset_dir, "val_dataset_features"))        

    @property
    def label_map(self) -> Dict[int, str]:
        """
            Return a dict mapping from id to actual word value
        """
        file_path = Path(self.file_manager.ms_dataset_description_dir, self.classes_description_json_name)
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find file with classes description: {str(file_path)}")

        # Open and load json
        with open(str(file_path)) as file:
            json_dict = json.load(file)

        return {i : w for (i,w) in enumerate(json_dict)}


    def download_train_dataset(self):
        """
            download videos for the training dataset
        """
        # Create if not exists
        p = Path(self.train_dataset_dir)
        if not p.exists():
            p.mkdir(parents=True)

        # Start download
        yt = YoutubeDownloader(self.train_dataset_dir)
        train_dataset_desc = self.read_train_dataset()
        yt.download([x.url for x in train_dataset_desc], filenames=[x.file + ".mp4" for x in train_dataset_desc])

    def download_test_dataset(self):
        """
            download videos for the test dataset
        """
        # Create if not exists
        p = Path(self.test_dataset_dir)
        if not p.exists():
            p.mkdir(parents=True)

        # Start download
        yt = YoutubeDownloader(self.test_dataset_dir)
        test_dataset_desc = self.read_test_dataset()
        yt.download([x.url for x in test_dataset_desc], filenames=[x.file + ".mp4" for x in test_dataset_desc])

    def download_val_dataset(self):
        """
            download videos for the training dataset
        """
        # Create if not exists
        p = Path(self.val_dataset_dir)
        if not p.exists():
            p.mkdir(parents=True)

        # Start download
        yt = YoutubeDownloader(self.val_dataset_dir)
        val_dataset_desc = self.read_val_dataset()
        yt.download([x.url for x in val_dataset_desc], filenames=[x.file + ".mp4" for x in val_dataset_desc])

    def _read_dataset_description(self, dataset_description_json_path : str) -> List[SignDescription]:
        """
            Read a list of sign descriptions from a dataset description, where the dataset
            description is in a json specified by the provided path relative to the microsoft 
            dataset local folder
        """

        # check path exists
        path_to_json = Path(self.file_manager.ms_dataset_description_dir, dataset_description_json_path)
        if not path_to_json.exists():
            raise FileNotFoundError(f"There's no file named '{dataset_description_json_path}' inside the local dir '{self.file_manager.ms_dataset_description_dir}'")

        with path_to_json.open("r") as js:
            json_list = json.load(js)

        # Check file consistency 
        if not isinstance(json_list, list):
            raise ValueError(f"The file {dataset_description_json_path} is not a valid dataset description, it does not contains a list of objects")

        
        return [ SignDescription.from_dict(d) for d in json_list ]

    def read_val_dataset(self) -> List[SignDescription]:
        """
            Return a list of sign description from the validation dataset
        """
        return self._read_dataset_description(str(Path(self.file_manager.ms_dataset_description_dir, self.val_dataset_name)))
    
    def read_train_dataset(self) -> List[SignDescription]:
        """
            Return a list of sign description from the training dataset
        """
        return self._read_dataset_description(str(Path(self.file_manager.ms_dataset_description_dir, self.train_dataset_name)))

    def read_test_dataset(self) -> List[SignDescription]:
        """
            Return a list of sign description from the test dataset
        """
        return self._read_dataset_description(str(Path(self.file_manager.ms_dataset_description_dir, self.test_dataset_name)))

    def _create_numeric_dataset(self, root_dir : str, video_datasets_dir : str, model : Holistic, description : List[SignDescription], skip_if_in_index : bool = True, display_vids : bool = False):
        """
            Create a numeric dataset in local storage from a dir with videos
        """
        numeric_dataset = NumericDatasetClient(root_dir, video_datasets_dir)
        numeric_dataset.save_sign_features_from_description(description, model, display_vids, skip_if_in_index  )

    @property
    def val_numeric_dataset_client(self) -> NumericDatasetClient:
        """
            Return a numeric dataset client that manages validation datasets
        """
        return NumericDatasetClient(self.val_numeric_dataset_dir, self.val_dataset_dir)

    @property
    def train_numeric_dataset_client(self) -> NumericDatasetClient:
        """
            Return a numeric dataset client that manages training datasets
        """
        return NumericDatasetClient(self.train_numeric_dataset_dir, self.train_dataset_dir)

    @property
    def test_numeric_dataset_client(self) -> NumericDatasetClient:
        """
            Return a numeric dataset client that manages testing datasets
        """
        return NumericDatasetClient(self.test_numeric_dataset_dir, self.test_dataset_dir)


    def create_val_numeric_dataset(self, model : Holistic, skip_if_in_index : bool = True, display_vids : bool = False):
        """
            Create a numeric validation dataset in local sotrage from the validation videos dir
        """
        # Check existence, and create if not exists
        path = Path(self.val_numeric_dataset_dir)
        if not path.exists():
            path.mkdir(parents=True)
        
        # load descriptions
        descriptions = self.read_val_dataset()

        self._create_numeric_dataset(str(path), self.val_dataset_dir, model, descriptions, skip_if_in_index, display_vids)

    def create_train_numeric_dataset(self, model : Holistic, skip_if_in_index : bool = True, display_vids : bool = False):
        """
            Create a numeric train dataset in local sotrage from the validation videos dir
        """
        # Check existence, and create if not exists
        path = Path(self.train_numeric_dataset_dir)
        if not path.exists():
            path.mkdir(parents=True)
        
        # load descriptions
        descriptions = self.read_train_dataset()

        self._create_numeric_dataset(str(path), self.train_dataset_dir, model, descriptions, skip_if_in_index, display_vids)

    def create_test_numeric_dataset(self, model : Holistic, skip_if_in_index : bool = True, display_vids : bool = False):
        """
            Create a numeric test dataset in local sotrage from the validation videos dir
        """
        # Check existence, and create if not exists
        path = Path(self.test_numeric_dataset_dir)
        if not path.exists():
            path.mkdir(parents=True)
        
        # load descriptions
        descriptions = self.read_test_dataset()

        self._create_numeric_dataset(str(path), self.test_dataset_dir, model, descriptions, skip_if_in_index, display_vids)



