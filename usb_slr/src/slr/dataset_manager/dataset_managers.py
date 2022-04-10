"""
    This file contains the necessary objects and operations 
    to manage datasets 
"""

# Python imports
import dataclasses
import json
from typing import Any, Dict, List, Optional, Type
from typing_extensions import Self 
from pathlib import Path

# Local imports
from slr.local_files.file_manager import FileManager

@dataclasses.dataclass
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
    box         : List[float]
    review      : Optional[int]

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

class DatasetManager:
    """
        Use this class to run common operations for datasets stored locally
    """

    def __init__(self, file_manager : FileManager = FileManager()):
        self._file_manager = file_manager
        pass

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

    