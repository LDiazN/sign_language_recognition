"""
    Executable file, write all click functions here.
"""

# Python imports
from ast import expr_context
from codecs import ignore_errors
from pathlib import Path
from typing import Optional

# Local imports 
from slr.config import settings
from slr.local_files.file_manager import FileManager
from slr.dataset_manager.dataset_managers import MicrosoftDatasetManager, PeruDatasetManager

# Third party imports
import click


@click.group()
def usb_slr():
    """
        Gloss-level Sign language recognition app
    """
    # Create folder if not exists
    home_path = Path(settings.slr_home)
    if not home_path.exists():
        click.echo(f"Creating home path in: {home_path}")
        home_path.mkdir()
    

@usb_slr.command()
@click.argument("dataset", nargs=1, required=True)
def fetch(dataset : str):
    """
        Fetch datasets. 
    """
    mngr = MicrosoftDatasetManager()

    if dataset == 'train':
        mngr.download_train_dataset() 
    elif dataset == 'test':
        mngr.download_test_dataset() 
    elif dataset == 'val':
        mngr.download_val_dataset() 
    else:
        click.echo(f"Invalid dataset name: {dataset}.") 
    
@usb_slr.group()
def generate_numeric():
    """Generate numeric dataset
    """

@generate_numeric.command()
@click.argument("dataset", nargs=1, required=True)
@click.option("--display", default=False, help="Display video processing, showing the video itself and the skeletal mesh")
def ms_dataset(dataset : str, display : bool = False):
    """
        Generate a numeric dataset and store it locally. The specified dataset should be one of 
        "train", "test", "eval"
    """

    # Consistency check
    valid_datasets = ["train", "test", "eval"]
    if dataset not in valid_datasets:
        click.echo(f"Bad dataset '{dataset}'. Choices are: {valid_datasets}")
        return 1

    from mediapipe.python.solutions.holistic import Holistic
    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        ds_manager = MicrosoftDatasetManager()
        funcs = {
            "train" : ds_manager.create_train_numeric_dataset,
            "test"  : ds_manager.create_test_numeric_dataset,
            "eval"  : ds_manager.create_val_numeric_dataset
        }
        funcs[dataset](holistic, display_vids=display) 

@generate_numeric.command()
@click.option("--display", default=False, help="Display video processing, showing the video itself and the skeletal mesh")
def peru_dataset(display : bool = False):
    """Generate a numeric dataset and store it locally
    """
    from mediapipe.python.solutions.holistic import Holistic
    with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        ds_manager = PeruDatasetManager()
        ds_manager.create_numeric_dataset(holistic, display_vids=display)

@generate_numeric.command()
@click.option("--display", default=False, help="Display video processing, showing the video itself and the skeletal mesh")
def argentina_dataset(display : bool = False):
    """Generate a numeric dataset and store it locally
    """
@usb_slr.group()
def load():
    """
        Load datasets
    """

@load.command()
@click.argument("path", nargs=1, required=True)
def ms(path : str):
    """
        Load a MS-ASL American Sign Language Dataset. More about such dataset here:
        https://www.microsoft.com/en-us/download/details.aspx?id=100121

        Arguments:
            - path : str = path to a zip file containing the MS-ASL American Sign Language Dataset
    """
    client = CLIClient()

    try:
        client.file_manager.store_ms_dataset(path)
    except ValueError as e:
        click.echo(f"Could not load given dataset '{path}'. Error: {e}", err=True)

@load.command()
@click.argument("path", nargs=1, required=True)
def peru(path : str):
    """
        Load a zip containing videos for the peru sign language dataset used for this research project

        Arguments:
            - path : str = path to a zip file containing videos for the peruvian dataset
    """
    client = CLIClient()

    try:
        client.file_manager.store_peru_dataset(path)
    except ValueError as e:
        click.echo(f"Could not load given dataset '{path}'. Error: {e}", err=True)

@load.command()
@click.argument("path", nargs=1, required=True)
def argentina(path : str):
    """
        Load a zip containing videos for the argentina sign language dataset used for this research project

        Arguments:
            - path : str = path to a zip file containing videos for the argentinavian dataset
    """
    client = CLIClient()

    try:
        client.file_manager.store_argentina_dataset(path)
    except ValueError as e:
        click.echo(f"Could not load given dataset '{path}'. Error: {e}", err=True)


@usb_slr.command()
def train(
    dataset : str, 
    output_dir : str, 
    profile_model : bool = False, 
    num_epochs : int = 2000, 
    num_folds : int = 6, 
    cache_dir : Optional[str] = None, 
    num_classes : Optional[int] = None,
    num_frames : Optional[int] = None,
    use_mobilenet : bool = False
    ):
    """Run a training with the specified configuration
    """

    # Sanity check
    valid_datasets = ["ms", "peru", "lsa64"]
    if dataset not in valid_datasets:
        click.echo(f"Invalid dataset: {dataset}, available options: {', '.join(valid_datasets)}", err=True)
        return
    
    # Set up num_classes, if not provided, take a default depending on selected dataset. 
    if num_classes is None:
        num_classes = {
            "ms" : 20,
            "lsa64" : 32,
            "peru" : 5
        }[dataset]

    # We do the same with num_frames as we did with num_classes,
    # we perform a sanity check and then we select the valid default if required 
    if num_frames is not None and num_frames <= 0:
        click.echo(f"Invalid number of frames: {num_frames}", err = True)

    if num_frames is None:
        num_frames = {
            "ms" : 60,
            "peru" : 80,
            "lsa64" : 200
        }[dataset]

    # Try to create output dir
    output_dir_path = Path(output_dir)
    try:
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True)
    except Exception as e:
        click.echo(f"Could not create output path dir: {output_dir_path}. Error: {e}", err= True)
    
    # Check cache dir 
    if cache_dir is not None:

        # If file exists but is not dir, raise an error
        # If dir does not exists, create it
        cache_dir_path = Path(cache_dir)
        if (cache_dir_path.exists() and not cache_dir_path.is_dir()):
            click.echo(f"Can't store cache in non-dir file: {cache_dir}", err = True)
            return
        elif not cache_dir_path.exists():
            try:
                cache_dir_path.mkdir(parents=True)
            except Exception as e:
                click.echo(f"Could not create cache dir: {cache_dir}. Error: {e}", err=True)
                return
    
    




class CLIClient:
    """
        Manage common CLI operations
    """

    def __init__(self) -> None:
        self._file_manager = FileManager()
        
    @property
    def file_manager(self):
        """
            File manager object to operate and query local files
        """
        return self._file_manager

