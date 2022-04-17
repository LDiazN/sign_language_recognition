"""
    Download youtube videos from a list of videos and store them locally
"""

# Python imports 
from ast import Set
from pathlib import Path
from typing import List, Optional
import logging
import pickle


# Third party imports
from termcolor import colored
from pytube import YouTube
from pytube.exceptions import VideoPrivate, VideoUnavailable
import tqdm

class YoutubeDownloader:
    """
        Manage youtube downloads. 

        # Parameters:
            - downloads_dir : `str` = Directory where to save files and filedata
    """

    def __init__(self, downloads_dir : str):
        # logging.basicConfig(level=logging.DEBUG)  # TODO organizar logging

        # Consistency check
        if not Path(downloads_dir).exists():
            raise ValueError(f"Provided path to downloads does not exists. Path: {downloads_dir}")

        self._downloads_dir = downloads_dir

        # Set of downloaded urls
        self._download_history = set({})

    @property
    def _history_file_name(self) -> str:
        """
            Name of history file 
        """
        return ".history.pkl"

    def download(self, urls : List[str], double_check_history : bool = False): # TODO implement double checking 
        """
            Download urls from provided list of urls

            # Parameters
                - urls : `[str]` = list of urls to youtube videos to scrape
                - double_check_history : `bool` = Double check that files in history are actually downloaded, 
                    and download them if not
        """

        logging.info(colored("Checkinf if history file is available...", 'cyan'))
        history_file = self._find_history_file()

        # If available, use that history
        if history_file:
            logging.info("Resuming from history file.")
            self._download_history = history_file
        else: # if not, start from scratch
            logging.info(colored("No history file found, downloading all videos.", 'cyan'))

        # Download videos
        logging.info(colored(f'Starting download for {len(urls)} urls', 'cyan'))
        for url in tqdm.tqdm(urls):

            # If already downloaded, skip it
            if url in self._download_history:
                logging.info(colored(f"Skipping video already downloaded according history: {url}", 'cyan'))
                continue

            # Try to download video
            try:
                self._get_video(url)
                self._download_history.add(url) # type: ignore
                self.save_history()
            except (VideoPrivate, VideoUnavailable) as e:
                logging.error(colored(f"Could not download video '{url}'. Error: {e}", "red"))
                self._download_history.add(url) # Add history when some of this error happens
                self.save_history()
        
        logging.info(colored(f"Download finished. {len(self._download_history)} videos succesfully downloaded")) # type: ignore

    def save_history(self):
        """
            Try to save the history file
        """
        path = Path(self._downloads_dir, self._history_file_name)

        with path.open("wb") as f:
            pickle.dump(self._download_history, f)

    
    def _get_video(self, url : str, out_path : Optional[str] = None, resolution : str = "360p", format : str = "mp4"):
        """
            Download video in url 'url' to path 'out_path' by resolution 'resolution' with format 'format' (mp4, wav) and 'fps'
            and return video filename in local storage
        """

        out_path = out_path or self._downloads_dir
        yt = YouTube(url)

        # Download video
        return yt.streams.filter(file_extension=format).get_by_resolution(resolution).download(out_path)



    def _find_history_file(self) -> Optional[set]:
        """
            Try to search for a file with a set of 
            already downloaded videos in the download dir

            # Returns
                None if no history file was found, a set with the history if found
        """
        
        path = Path(self._downloads_dir, self._history_file_name)

        # Check if file exists
        if not path.exists():
            return None

        # Try to load pickle object
        with path.open('rb') as f:
            history = pickle.load(f)
        
        # Check if is actualy a set of urls
        if not isinstance(history, set):
            raise ValueError(f"Invalid history file, should contain a set, but it contains: {type(history)}")

        return history
        
