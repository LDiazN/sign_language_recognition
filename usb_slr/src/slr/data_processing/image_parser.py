"""
    This class will parse an image to compute the skeletal annotations
"""

# Python imports
from turtle import right
from typing import Iterable, List, NamedTuple, Tuple
import dataclasses
from typing_extensions import Self
import enum

# Third party imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing
from scipy.misc import face

class EdgeType(enum.Enum):
    """
        Possible variations of a type of pose
    """
    POSE = 0
    FACE = 1
    LEFT_HAND = 2
    RIGHT_HAND = 3

@dataclasses.dataclass
class Edge:
    """
        An edge description, start position, end position and type of edge 
    """
    start : int
    end : int
    type : EdgeType

    @property
    def as_array(self) -> np.ndarray:
        """
            An array representation of this edge with the following format:
                [start, end, type]
        """
        return np.array([self.start, self.end, self.type.value])

@dataclasses.dataclass
class JointData:
    """
        Data for a joint, position and visibility
    """
    x : float 
    y : float
    z : float
    visibility : bool

    @property
    def as_array(self) -> np.ndarray:
        """
            array representation for this joint
        """
        return np.array([self.x, self.y, self.z, int(self.visibility)])

@dataclasses.dataclass
class PoseGraph:
    """
        A graph representing a pose
    """
    edges : List[Edge]
    n_nodes : int 
    joint_data : List[JointData]

    @property
    def edges_array(self) -> np.ndarray:
        """
            Array representing the edges
        """
        return np.array([e.as_array for e in self.edges])

    @property
    def joints_array(self) -> np.ndarray:
        """
            Array representation for joints
        """
        return np.array([j.as_array for j in self.joint_data])

@dataclasses.dataclass
class PoseValues:
    """
        Represents all values for a pose
    """

    # Holds both positions and visibility (x,y,z, visibility), vertices 11 and 12 are shoulder joints
    pose : np.ndarray # Size: 132

    # Holds positions (x,y,z)
    face : np.ndarray # Size: 1404
    left_hand : np.ndarray # Size: 63
    right_hand : np.ndarray # Size: 63

    @classmethod
    def from_mediapipe_result(cls, results : NamedTuple):
        """
            Create a pose value objects from a mediapipe results object
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132) # type: ignore 
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404) # type: ignore 
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) # type: ignore 
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) # type: ignore 

        return cls(pose=pose, face=face, left_hand = lh, right_hand = rh)
    
    @property
    def concatenated(self) -> np.ndarray:
        """
            Return a single array with all numbers in a single row.
            Array order:
            pose, face, left_hand, right_hand
        """

        return np.concatenate([self.pose, self.face, self.left_hand, self.right_hand])

    def normalize_location(self):
        """
            Perform a location normalization. It means that the entire pose will be at the origin, 
            it does not matter in which part of the screen it originally was. 
        """
        pose_vertices = [np.array(x) for x in self.pose_vectors]
        face_vertices = [np.array(x) for x in self.face_vectors]
        lhand_vertices = [np.array(x) for x in self.left_hand_vectors]
        rhand_vertices = [np.array(x) for x in self.right_hand_vectors]

        # Shoulder joints
        shoulder_1 = pose_vertices[11][:3]
        shoulder_2 = pose_vertices[12][:3]

        pivot = .5 * (shoulder_1 + shoulder_2) 
        face_vertices = [v - pivot for v in face_vertices]
        lhand_vertices = [v - pivot for v in lhand_vertices]
        rhand_vertices = [v - pivot for v in rhand_vertices]
        pivot = np.array([*pivot, 0])
        pose_vertices = [v - pivot for v in pose_vertices]

        self.face = np.array(face_vertices).flatten()
        self.pose = np.array(pose_vertices).flatten()
        self.left_hand = np.array(lhand_vertices).flatten()
        self.right_hand = np.array(rhand_vertices).flatten()



    def display(self):
        """
            Display this pose as a graph in matplot lib
        """
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        points = [(x,-y) for (x,y,_,_) in self.pose_vectors] + [(x,-y) for (x,y,_) in self.face_vectors]
        xs = [x for (x,_) in points]
        ys = [y for (_,y) in points]

        maxi_x = max(max(xs), abs(min(xs)))
        maxi_y = max(max(ys), abs(min(ys)))
        maxi = max(maxi_x, maxi_y)

        plt.axis([-maxi, maxi, -maxi, maxi])
        plt.plot(xs, ys, 'r*')
        plt.show()

    @property
    def pose_vectors(self) -> Iterable[Tuple[float, float, float, float]]:
        """
            Returns the vectors that correspond to the pose, in the following format:
                (x,y,z, visibility)
        """
        for i in range(0, len(self.pose), 4):
            yield (self.pose[i], self.pose[i + 1], self.pose[i+2], self.pose[i+3])
    
    @property
    def face_vectors(self) -> Iterable[Tuple[float, float, float]]:
        """
            Returns the vectors that correspond to the face, in the following format:
                (x,y,z)
        """
        for i in range(0, len(self.face), 3):
            yield (self.face[i], self.face[i + 1], self.face[i+2])

    def _hand_vectors(self, hand : np.ndarray) -> Iterable[Tuple[float, float, float]]:
        """
            Returns the vectors that correspond to the hand, in the following format:
                (x,y,z)
        """
        for i in range(0, len(hand), 3):
            yield (hand[i], hand[i + 1], hand[i+2])

    @property
    def left_hand_vectors(self) -> Iterable[Tuple[float, float, float]]:
        """
            Returns the vectors that correspond to the left hand, in the following format:
                (x,y,z)
        """
        for x in self._hand_vectors(self.left_hand):
            yield x

    @property
    def right_hand_vectors(self) -> Iterable[Tuple[float, float, float]]:
        """
            Returns the vectors that correspond to the right hand, in the following format:
                (x,y,z)
        """
        for x in self._hand_vectors(self.right_hand):
            yield x


    @classmethod
    def from_array(cls, array : np.ndarray) -> Self:
        """
            Return the Pose value from an array
        """
        assert array.shape == (132 + 1404 + 63 + 63,)

        pose = array[:132]
        face = array[132:132 + 1404]
        left_hand = array[132 + 1404:1404 + 132 + 63]
        right_hand = array[1404 + 132 + 63 : 1404 + 132 + 63 + 63]

        return cls(pose, face, left_hand, right_hand)
    
    @property
    def as_graph(self) -> PoseGraph:
        """
            A graph representation for this pose. Useful with GNNs
        """

        joint_datas = []
        joint_counts = {
            "pose" : 0,
            "face" : 0,
            "lh" : 0,
            "rh" : 0
        }

        # Parse pose values, x, y, z and visibility
        end = len(self.pose)
        for i in range(0, end, 4):
            pose = JointData(self.pose[i], self.pose[i+1], self.pose[i + 2], self.pose[i + 3])
            joint_datas.append(pose)
            joint_counts["pose"] += 1


        data = [(self.face, "face"), (self.left_hand, "lh"), (self.right_hand, "rh")]
        for (d, name) in data:
            end = len(d)
            for i in range(0, end, 3):
                x, y, z = self.face[i], self.face[i+1], self.face[i+2]
                vis = x == y == z
                joint = JointData(x,y,z,vis)
                joint_datas.append(joint)
                joint_counts[name] += 1
        
        # Generate edges information. 
        #  - pose nodes are in [0, pose)
        #  - face nodes are in [pose, pose + face)
        #  - left hand nodes are in [pose + face, pose + face + lh)
        #  - right hand nodes are in [pose + face + lh, pose + face + lh + rh)
         
        # Add pose nodes
        edges = []
        for (start_node, end_node) in POSE_CONNECTIONS:
            edges.append(Edge(start_node, end_node, EdgeType.POSE))

        # Add face nodes
        for (start_node, end_node) in FACEMESH_CONTOURS:
            start_node += joint_counts['pose']
            end_node += joint_counts['pose']
            edges.append(Edge(start_node, end_node, EdgeType.FACE))

        # Add left hand nodes
        for (start_node, end_node) in HAND_CONNECTIONS:
            start_node += joint_counts['pose'] + joint_counts['face']
            end_node += joint_counts['pose'] + joint_counts['face']
            edges.append(Edge(start_node, end_node, EdgeType.LEFT_HAND))

        # Add left hand nodes
        for (start_node, end_node) in HAND_CONNECTIONS:
            start_node += joint_counts['pose'] + joint_counts['face'] + joint_counts['lh']
            end_node += joint_counts['pose'] + joint_counts['face'] + joint_counts['lh']
            edges.append(Edge(start_node, end_node, EdgeType.RIGHT_HAND))
            
        return PoseGraph(edges, sum(joint_counts.values()), joint_datas)


class ImageParser:
    """
        Use an image parser to compute the skeletal graph 
        from an image or video
    """
    
    def __init__(self, model : mp_holistic.Holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)) -> None:
        self._model = model
        
    @property
    def model(self) -> mp_holistic.Holistic:
        """
            Model used for predictions (computing skeletal graph)
        """
        return self._model

    def mediapipe_detection(self, image : np.ndarray, draw_landmarks : bool = False) -> Tuple[np.ndarray, NamedTuple]: # TODO annotate return type
        """
            Get the mediapipe prediction for this image.

            # Parameters
                - image : `np.ndarray` = image in BGR opencv format
                - draw_landmarks : bool = (optional) Drawl debug landmarks. Useful for debug
            # Return
                Image, Mediapipe results
        """
        
        # Color conversion 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store previous flag
        old_flag = image.flags.writeable
        # Set as non writable now
        image.flags.writeable = False

        # Compute skeletal graph
        results = self.model.process(image)

        # Restore writability
        image.flags.writeable = old_flag

        # Convert back to original format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if draw_landmarks:
            self._draw_landmarks(image, results)

        return image, results

    def parse_image(self, image : np.ndarray, draw_landmarks : bool = False) -> Tuple[np.ndarray, PoseValues]:
        """
            Extract pose values from an image. 
            You can optionally draw landmarks in the image, useful for debugging purposes.

            # Returns 
                the image and the pose values extracted for this image
        """
        image, results = self.mediapipe_detection(image, draw_landmarks)

        return image, PoseValues.from_mediapipe_result(results)

    def _draw_landmarks(self, image : np.ndarray, results : NamedTuple):
        """
            Draw skeletal landmarks in provided image. Provided results should match
            provided image, or the results will be inconsistent
        """
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, # type: ignore 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, # type: ignore 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, # type: ignore 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, # type: ignore 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def run_from_webcam(self):
        """
            Run a parsing from webcam, useful for testing purposes
        """
        cap = cv2.VideoCapture(0)

        # Set mediapipe model
        while cap.isOpened():

            # Read from feed
            ret, frame = cap.read()

            img, res = self.parse_image(frame, True)

            # Display to screen
            cv2.imshow("Body landmarks", img)

            # Break if requested 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Free resources
        cap.release()
        cv2.destroyAllWindows()
