"""
    Executable file, write all click functions here.
"""

# Python imports
from pathlib import Path

# Local imports 
from slr.config import settings
from slr.local_files.file_manager import FileManager
from slr.dataset_manager.dataset_managers import DatasetManager

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
    mngr = DatasetManager()

    if dataset == 'train':
        mngr.download_train_dataset() 
    elif dataset == 'test':
        mngr.download_test_dataset() 
    elif dataset == 'val':
        mngr.download_val_dataset() 
    else:
        click.echo(f"Invalid dataset name: {dataset}.") 
    

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

