"""
    This class will parse an image to compute the skeletal annotations
"""

# Python imports
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
import dgl
import torch

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

    @classmethod
    def n_channels(cls) -> int:
        """How many information channels does this data have. 4 for now, x,y,z, visibility

        Returns:
            int: Amount of information channels
        """
        return len(dataclasses.fields(cls))

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

    @property
    def as_dgl_graph(self) -> dgl.DGLGraph:
        """Return a DGLGraph representation of this pose

        Returns:
            dgl.DGLGraph: DGL graph representation of this pose
        """

        # Create graph by providing list of edges and node count
        graph = dgl.graph(([ e.start for e in self.edges ], [e.end for e in self.edges]), num_nodes=self.n_nodes)

        # Add data for joints as nodes data
        graph.ndata['pos_and_vis'] = torch.tensor( [[d.x, d.y, d.z, d.visibility] for d in self.joint_data])

        # Add edge type as data for edges
        graph.edata['type'] = torch.tensor([e.type.value for e in self.edges]).float()

        # Add reverse edges as this graph is undirected
        graph = dgl.add_reverse_edges(graph)
        graph = dgl.add_self_loop(graph)

        return graph

    @classmethod
    def get_expected_joint_amount(cls, include_face : bool = True) -> int:
        """Get the expected amount of joints. Some joint data might be missing if not visible

        Returns:
            int: amount of joints expected for the pose graph
        """
        return 543 - int(not include_face) * 1404//3

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
    
    def concatenated(self, exclude_face : bool = False) -> np.ndarray:
        """
            Return a single array with all numbers in a single row.
            Array order:
            pose, face, left_hand, right_hand
        """

        return np.concatenate([self.pose, self.face if not exclude_face else np.array([]), self.left_hand, self.right_hand])

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
    
    def as_graph(self, include_face : bool = True) -> PoseGraph:
        """Return a graph representation for this pose 

        Args:
            include_face (bool, optional): If should include face joints. Defaults to True.

        Returns:
            PoseGraph: A graph version of this pose
        """

        # Parse joint data as arrays
        # IT'S IMPORTANT TO KEEP THE ORDER OF THE SUBGRAPHS IN SUBSEQUENT OPERATIONS
        pose_joint_data, lh_joints_data, rh_joints_data, face_joints_data = self._get_joints_datas(include_face)
        
        joint_data = [pose_joint_data, lh_joints_data, rh_joints_data, face_joints_data]
        joint_maps = [[], [], [], []]
        
        # Compute node mapping for each node 
        next_node_id = 0
        for (l, jmap) in zip(joint_data, joint_maps):
            for _ in l:
                jmap.append(next_node_id)
                next_node_id += 1

        # Create edgelist: Every subgraph has its own edgelist with its local node numbering. We have to take edges
        # from there but remap them so the node numbering matches our node map
        edges = []
        edge_types = [EdgeType.POSE, EdgeType.LEFT_HAND, EdgeType.RIGHT_HAND, EdgeType.FACE]
        for (conns, joint_map, edge_type) in zip([POSE_CONNECTIONS, HAND_CONNECTIONS, HAND_CONNECTIONS, FACEMESH_CONTOURS if include_face else []], joint_maps, edge_types):
            for (start, end) in conns:
                edges.append(Edge(joint_map[start], joint_map[end], edge_type))
        
        # Create graph now that we have edges and joint features
        joint_data = [*pose_joint_data, *lh_joints_data, *rh_joints_data, *face_joints_data]
        return PoseGraph(edges, len(joint_data), joint_data)


    def _get_joints_datas(self, include_face : bool) -> Tuple[List[JointData],List[JointData], List[JointData], List[JointData]]:
        """
            return a tuple with a list in each coordinate corresponding to the data corresponding to the joints of 
            (pose, left hand, right hand, face). If include_face == false, face will be empty
        """
        # Parse arrays as lists of joint data
        pose_joint_data = [
                JointData( 
                    self.pose[i], 
                    self.pose[i+1], 
                    self.pose[i+2], 
                    self.pose[i+3]
                ) 
                for i in range(0, len(self.pose), 4)
            ]
        
        body_parts = [self.left_hand, self.right_hand, self.face if include_face else []]
        results = []
        for bp in body_parts:
            new_joints = [
            JointData(
                    bp[i],
                    bp[i+1],
                    bp[i+2],
                    not (bp[i] == bp[i+1] == bp[i+2] == 0.)
                ) 
                for i in range(0, len(bp), 3)
            ]

            results.append(new_joints)

        lh_joints_data, rh_joints_data, face_joints_data = results

        return pose_joint_data, lh_joints_data, rh_joints_data, face_joints_data
    

    @property
    def node_count(self) -> int:
        """
            Return amount of nodes in this graph
        """
        return len(self.pose) // 4 + len(self.left_hand) // 3 + len(self.right_hand) // 3 + len(self.face) // 3


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

    def mediapipe_detection(self, image : np.ndarray, draw_landmarks : bool = False) -> Tuple[np.ndarray, NamedTuple]: 
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
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, # type: ignore 
        #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
        #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                         ) 
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
