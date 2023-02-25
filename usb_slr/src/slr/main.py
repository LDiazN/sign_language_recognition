"""
    Executable file, write all click functions here.
"""

# Python imports
from pathlib import Path
from typing import Optional

# Local imports 
from slr.config import settings
from slr.local_files.file_manager import FileManager
from slr.dataset_manager.dataset_managers import MicrosoftDatasetManager, PeruDatasetManager

# Third party imports
import click
from traitlets import default


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
@click.argument("dataset", nargs = 1, required = True)
@click.argument("output_dir", nargs = 1, required = True)
@click.option("--profile", default = False, help="If should profile model training; might be expensive in memory and time")
@click.option("--epochs", default = 2000, help = "Amount of epochs to train per fold")
@click.option("--folds", default = 6, help = "Amount of folds for training")
@click.option("--cache_dir", default = None, help = "Dir where to store cache for training")
@click.option("--classes", default = None, help = "How many classes to use for the specified dataset. Might raise an error if greater than its corresponding class count")
@click.option("--frames", default = None, help = "How many frames to use per sign")
@click.option("--mobilenet", default = False, help = "If should use mobilenet instead of custom CNN module")
@click.option("--experiment_name", default = "tmlstm_experiment", help = "Experiment name, used for plots and files")
@click.option("--wandb", default = False, help = "If should use wandb to monitor and register training")
def train(
    dataset : str, 
    output_dir : str, 
    profile : bool = False, 
    epochs : int = 2000, 
    folds : int = 6, 
    cache_dir : Optional[str] = None, 
    classes : Optional[int] = None,
    frames : Optional[int] = None,
    mobilenet : bool = False,
    experiment_name : str = "tmlstm_experiment",
    wandb : bool = False
    ):
    """Run a training with the specified configuration.
        Possible values for DATASET: ms, peru, lsa64 
    """

    # Sanity check
    valid_datasets = ["ms", "peru", "lsa64"]
    if dataset not in valid_datasets:
        click.echo(f"Invalid dataset: {dataset}, available options: {', '.join(valid_datasets)}", err=True)
        return
    
    # Set up num_classes, if not provided, take a default depending on selected dataset. 
    if classes is None:
        classes = {
            "ms" : 20,
            "lsa64" : 32,
            "peru" : 5
        }[dataset]

    # We do the same with num_frames as we did with num_classes,
    # we perform a sanity check and then we select the valid default if required 
    if frames is not None and frames <= 0:
        click.echo(f"Invalid number of frames: {frames}", err = True)

    if frames is None:
        frames = {
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
    else:
        cache_dir_path = Path(settings.slr_cache)
        try:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
                click.echo(f"Could not create cache dir specified in environment: {cache_dir}. Error: {e}", err=True)
                return
        
    
    # Import models here to avoid slow start up for ML requirements loading
    from slr.model.trajectory_map_lstm_model import TMLSTMClassifierTrainer
    trainer = TMLSTMClassifierTrainer(dataset, output_dir_path, profile, epochs, folds, cache_dir_path, classes, frames, experiment_name, mobilenet, wandb)
    trainer.run()




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

