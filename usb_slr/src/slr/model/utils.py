"""Common utilities used for training and inference
"""

# Python imports
import pickle
from typing import Any, Callable, Dict
from pathlib import Path

# Local imports
from slr.config import settings
class Cacher:
    """Class to manage cache for a given function output. If cache exists in disk, then use cached version 
        (might be out of date) instead of function output
    """

    def __init__(self, fn_to_cache : Callable[[], Any], cache_name : str = "cache", args : Dict[str, Any] = {}, cache_path : str = settings.slr_cache) -> None:
        cache_name = cache_name.split()[0]
        cache_name = f".{cache_name}.pkl"

        self._cache_name = cache_name
        self._path = Path(cache_path, cache_name)
        self._fn_to_cache = fn_to_cache

        self._args = args

    def get(self) -> Any:
        """
            Retrieve cached output if any, or call the cached function and return and save its output
        """ 

        # If exists, load it   
        if self.exists():
            with self._path.open("rb") as f:
                return pickle.load(f)

        # if not, call and save
        result = self._fn_to_cache(**self._args)
        with self._path.open("wb") as f:
            pickle.dump(result, f)
        
        return result

    def exists(self) -> bool:
        """Checks if the cache exists in disk

        Returns:
            bool: whether the scache exists or not
        """

        return self._path.exists()

def compute_output_shape(in_shape : int, padding : int, dilation : int, kernel_size : int, stride : int) -> int:
    """Compute side size for output of cnn layer

    Args:
        in_shape (int): Side len of input image
        kernel_size (int): Size of filters
        stride (int): Layer stride

    Returns:
        int: Resulting side len of output image
    """
    return (in_shape + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1

