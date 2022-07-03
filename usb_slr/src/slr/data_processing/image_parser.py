"""
    This class will parse an image to compute the skeletal annotations
"""

# Python imports
from ntpath import join
from typing import Callable, Iterable, List, NamedTuple, Tuple, Union
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
from typing import Optional
from pyparsing import Opt
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

    def as_cv_img(self, 
        width : int = 512, 
        height : int = 512, 
        show_edges : bool = False, 
        joint_radius_px : int = 3, 
        bounding_box : Optional[Tuple[float, float, float, float]] = None,
        joint_color : Optional[Tuple[float, float, float]] = None,
        joint_color_intensity : Optional[np.ndarray] = None
        ) -> np.ndarray:
        """Draw this graph in a CV image

        Returns:
            np.array: Resulting cv image
        """

        JOINT_COLOR = (1,0,0)
        EDGE_COLOR = (0,1,0)

        joint_color = np.array(joint_color or JOINT_COLOR)

        if joint_color_intensity is None:
            joint_color_intensity = np.ones(len(self.joint_data))

        if bounding_box:
            min_x, max_x, min_y, max_y = bounding_box
        else: 
            min_x, max_x, min_y, max_y = PoseGraph._bounding_box_from_graph(self)

        # Get aspect ratio
        original_h = max_y - min_y
        original_w = max_x - min_x
        aspect_ratio = original_h/original_w


        # Transform point to the correct coordinate system
        def to_img(x : float, y : float) -> Tuple[float, float]:
            # TODO maybe check for aspect ratio
            return (
                int((x - min_x)/(max_x - min_x) * width), 
                int((y - min_y)/(max_y - min_y) * height)
                )

        # Create black background
        img = np.zeros((width, height,3))

        # Draw points in image
        for (j, intensity) in zip(self.joint_data, joint_color_intensity):
            if not j.visibility:
                continue
            pos_img = to_img(j.x, j.y)
            cv2.circle(img, pos_img, joint_radius_px, intensity * joint_color, -1)


        if show_edges:
            for edge in self.edges:
                p1 = self.joint_data[edge.start]
                p2 = self.joint_data[edge.end]

                if not p1.visibility or not p2.visibility:
                    continue

                p1_img = to_img(p1.x, p1.y)
                p2_img = to_img(p2.x, p2.y)

                cv2.line(img, p1_img, p2_img, EDGE_COLOR, 2)

        return img

    @staticmethod
    def as_trajectory_cv_img(graphs : list, width : int = 512, height : int = 512, show_edges : bool = False, joint_radius_px : int = 3) -> np.ndarray:
        """Create an image with all frames of the given graph, such that earlier frames are darker than newer frames

        Args:
            graphs (list): List of graph to render

        Returns:
            np.ndarray: a cv2 image
        """

        pose_graphs : List[PoseGraph] = graphs

        # fing overall bounding box
        min_x, max_x, min_y, max_y = PoseGraph._bounding_box_from_sign(pose_graphs)
                
        # Create image for each graph
        imgs = (g.as_cv_img(width, height, show_edges, joint_radius_px, bounding_box=(min_x, max_x, min_y, max_y)) for g in pose_graphs)

        # Compute color incresing for each graph 
        n_frames = len(pose_graphs)
        increase = 1.0 / n_frames

        # Sum each img 
        result = np.zeros((width, height, 3))
        for (i, img) in enumerate(imgs):
            result += increase * (i + 1) *  img

        # Return img with all frames
        return result

    @staticmethod
    def sign_to_imgs(sign : list, width : int = 512, height : int = 512, show_edges : bool = False, joint_radius_px : int = 3) -> List[np.ndarray]:
        """Create a list of graph images from the given sign (list of graphs) 

        Args:
            sign (list): list of graphs specifying a sign
            width (int, optional): Width of resulting images. Defaults to 512.
            height (int, optional): Height of resulting images, in pixels. Defaults to 512.
            show_edges (bool, optional): If should show edges in each image. Defaults to False.
            joint_radius_px (int, optional): Radius for joints in the image. Defaults to 3.

        Returns:
            List[np.ndarray]: List of images. Will all be normalized to the same bounding box1
        """
        sign_graphs : List[PoseGraph] = sign
        bb = PoseGraph._bounding_box_from_sign(sign_graphs)
        return [g.as_cv_img(width=width, height=height, show_edges=show_edges, joint_radius_px=joint_radius_px, bounding_box=bb) for g in sign_graphs]

    @staticmethod
    def _bounding_box_from_graph(graph) -> Tuple[float, float, float, float]:
        """Compute bounding box for a given graph. Is just a bounding box that contains every joint

        Args:
            graph (PoseGraph): Graph where the joints will be taken from

        Returns:
            Tuple[float, float, float, float]: min_x, max_x, min_y, max_y
        """

        pose_graph : PoseGraph = graph

        min_x = float("inf")
        max_x = float("-inf")

        min_y = float("inf")
        max_y = float("-inf")
        for j in pose_graph.joint_data:

            # Ignore non visible joints
            if not j.visibility:
                continue
            
            min_x = min(min_x, j.x)
            max_x = max(max_x, j.x)
            min_y = min(min_y, j.y)
            max_y = max(max_y, j.y)

        return min_x, max_x, min_y, max_y
    
    @staticmethod
    def _bounding_box_from_sign(sign : list) -> Tuple[float, float, float, float]:
        """Returna bounding box containing all the joints in all signs

        Args:
            sign (List[PoseGraph]): List of PoseGraph defining a graph

        Returns:
            Tuple[float, float, float, float]: min_x, max_x, min_y, max_y
        """
        sign_graphs : List[PoseGraph] = sign

        # Generate bounding boxes for each graph
        bounding_boxs = (PoseGraph._bounding_box_from_graph(g) for g in sign_graphs)

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for (new_min_x, new_max_x, new_min_y, new_max_y) in bounding_boxs:
            min_x = min(min_x, new_min_x)
            max_x = max(max_x, new_max_x)

            min_y = min(min_y, new_min_y)
            max_y = max(max_y, new_max_y)

        return min_x, max_x, min_y, max_y

    @staticmethod
    def _mean_graph(sign : list):
        """Returns the mean graph of a sign: A graph such that the position of every joint is the average between the non-zero positions in a sign (seq of graphs)

        Args:
            sign (list): List of PoseGraphs defining a sign

        Returns:
            PoseGraph: Mean pose of a sign 
        """

        assert sign != [], "mean graph expects to be at the least one frame"

        mean_joints = PoseGraph.mean_positions(sign)
        graph = sign[0]

        # Update with new mean values
        for (j, r) in zip(graph.joint_data, mean_joints):
            j.x, j.y = r

        return graph

    @staticmethod
    def _non_zero_count(sign : list) -> np.ndarray:
        """Count how many non zero joints there is in each joint along a sign

        Args:
            sign (list): list of pose graph defining a sign

        Returns:
            np.ndarray: variance array per joint
        """
        non_zero_count = np.zeros((len(sign[0].joint_data), 2))

        for g in sign:
            # Build tensor of joints with sum of coordinates
            joint_tensor = np.array([[j.x, j.y] for j in g.joint_data])

            # Add non-zeros to compute final division
            non_zero_count += (joint_tensor != [0,0]).astype(int)

        return non_zero_count


    @staticmethod
    def mean_positions(sign : list) -> np.ndarray:
        """
            Compute mean positions of joints in sign
        """
        mean_joints = np.zeros((len(sign[0].joint_data), 2))
        # Use non zero count to ignore zeroed joints
        non_zero_count = PoseGraph._non_zero_count(sign)

        for g in sign:
            # Build tensor of joints with sum of coordinates
            joint_tensor = np.array([[j.x, j.y] for j in g.joint_data])

            # Sum values of joints to mean joints
            mean_joints += joint_tensor

        # Update joint values in this graph to mean values. Just to recycle edge data
        return mean_joints / non_zero_count


    @staticmethod
    def joint_variance(sign : list) -> np.ndarray:
        """Compute variance of joint positions in a given sign

        Args:
            sign (list): list of graphs defining frames in a sign

        Returns:
            np.ndarray: variance value per joint
        """

        assert sign 
        graph_list : List[PoseGraph] = sign

        mean_joints = PoseGraph.mean_positions(sign)
        non_zero_count = PoseGraph._non_zero_count(sign)

        # compute sum of (X_i - mean) ^ 2
        result = np.zeros_like(mean_joints)
        for g in graph_list:

            # Build joint positions array
            joints = np.array([[j.x, j.y] for j in g.joint_data])
            result += (joints - mean_joints) ** 2 

        non_zero_count = non_zero_count - 1

        return result / non_zero_count

    @staticmethod
    def variance_graph(sign : list) -> np.ndarray:
        """ Create a variance graph where the positions of nodes are mean positions, and 
            its color represents how much variance it has relative to other nodes

        Args:
            sign (list): A sign defined by a list of graphs

        Returns:
            np.ndarray: A CV image encoded as np array
        """

        mean_graph : PoseGraph = PoseGraph._mean_graph(sign)
        variance = np.mean(PoseGraph.joint_variance(sign), axis=1) 

        # Normalice variance with min max so we can use that value as intensity
        mini = np.min(variance)
        maxi = np.max(variance)
        variance = (1/(maxi - mini)) * (variance - mini)

        ORANGE = (0.0,165.0 / 255.0, 255.0 / 255.0)

        return mean_graph.as_cv_img(show_edges= True, joint_radius_px=5, joint_color=ORANGE, joint_color_intensity=variance)

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
