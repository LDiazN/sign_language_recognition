"""
    Use the class in this file to create a dataframe or csv
    from a set of videos. The dataframe will hold a 
    table with floats representing skeletal anotations in each video
"""

# Local imports
from typing import List
from typing_extensions import Self
from slr.data_processing.image_parser import ImageParser, PoseValues

# Python imports
import os
from pathlib import Path
import logging

# Third party imports
import cv2
from cv2.mat_wrapper import Mat
import numpy as np
from matplotlib import pyplot as plt
import time
from mediapipe.python.solutions.holistic import Holistic


class VideoConverter:
    """
        Convert from videos to skeletal annotations as a dataframe
    """
    def __init__(self, image_parser : ImageParser) -> None:
        
        self._image_parser = image_parser

    @classmethod
    def from_holistic_model(cls, model : Holistic) -> Self:
        return cls(ImageParser(model))

    @property
    def image_parser(self) -> ImageParser:
        """
            Return the image parser used by this image to get the skeletal annotations
        """
        return self._image_parser

    def parse_video(self, video : cv2.VideoCapture, start_second : float, end_second : float, display_video : bool = False) -> List[PoseValues]:
        """
            Parse pose values from provided image from start second to end second
        """
        assert end_second > start_second, "Inconsistent video bounds"

        # Get video fps count
        fps : float = video.get(cv2.CAP_PROP_FPS)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get initial and final frame index
        start_frame = int(start_second * fps)
        end_frame = int(end_second * fps)

        # Skip those frames 
        for _ in range(0, start_frame):
            video.read()
        
        # Parse remaining video
        parsed_data = []
        for _ in range(start_frame, end_frame):
            success, frame = video.read()

            # break if video ended
            if not success: break

            image, data = self.image_parser.parse_image(frame, display_video)
            parsed_data.append(data)

            # Show video if requested
            if display_video:
                # Display to screen
                cv2.imshow("Parsing video:", image)

                # Break if requested 
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        return parsed_data

    def parse_video_from_file(self, video_path : str, start_second : float, end_second : float, display_video : bool = False):
        """
            Parse data from video, but from file
        """
        path_object = Path(video_path)

        # Consistency check
        if not path_object.exists():
            raise FileNotFoundError(f"Provided video file could not be found: {video_path}")
        
        video = cv2.VideoCapture(video_path)

        return self.parse_video(video, start_second, end_second, display_video)