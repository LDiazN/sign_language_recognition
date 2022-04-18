"""
    Use the class in this file to create a dataframe or csv
    from a set of videos. The dataframe will hold a 
    table with floats representing skeletal anotations in each video
"""

# Local imports
from slr.data_processing.image_parser import ImageParser

# Python imports
import os

# Third party imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe import solutions


class DataConverter:
    """
        Convert from videos to skeletal annotations as a dataframe
    """
    def __init__(self, image_parser : ImageParser) -> None:
        
        self._image_parser = image_parser

    @property
    def image_parser(self) -> ImageParser:
        """
            Return the image parser used by this image to get the skeletal annotations
        """
        return self._image_parser