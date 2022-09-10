"""
(trys to) Implements real time SLR using the specified model and data processor
"""
# Local imports
from slr.data_processing.image_parser import ImageParser, PoseValues

# Third party imports 
import cv2
import termcolor as c
from mediapipe.python.solutions import holistic as mp_holistic



class RealTimeParser:
    """This object will try to parse the signs from a sign video. It's logic is quite simple, it will store
    the last N frames, then it will uniform-sample K of those frames and input them to the model. If no sign has a 
    confidence score grater than T, then no sign is reported. Otherwise, the highest-scoring sign is reported. 
    Read from webcam by default.
    """

    def __init__(
            self, 
            frames_to_store : int, 
            frames_to_sample : int, 
            confidence : float = 0.8, 
            video_capture : cv2.VideoCapture = cv2.VideoCapture(0), 
            holistic_model : mp_holistic.Holistic = mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5)
            ) -> None:
        assert frames_to_sample > 0
        assert frames_to_store > 0
        assert 0 < confidence < 1

        self._frames_to_store = frames_to_store
        self._frames_to_sample = frames_to_sample
        self._confidence = confidence
        self._video_capture = video_capture
        self._image_parser = ImageParser(holistic_model)
        self._stored_frames = []

    def run(self, show_landmarks : bool = True):
        """Runs a recognition loop
        """
        c.cprint("Starting real time recognition. Press Q to exit...", color = "blue")

        while True:
            ret, frame = self._video_capture.read()

            # Process image with holistic to retrieve PoseValues
            frame, pose_values =  self._image_parser.parse_image(frame, show_landmarks)

            self._add_new_frame_data(frame)

            cv2.imshow("Real Time Parser", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                c.cprint("Q pressed, Shutting down...", color='yellow')
                break
    
        c.cprint("Successfully finished real time recognition!", color = 'green')

    def _add_new_frame_data(self, pose_value : PoseValues):
        """Add a new frame of data to the stored frames. If greater than amount of frames to store, 
        remove one from the start

        Args:
            pose_values (PoseValues): Frame to add
        """
        self._stored_frames.append(pose_value)
        if len(self._stored_frames) > self._frames_to_store:
            self._stored_frames.pop(0)

