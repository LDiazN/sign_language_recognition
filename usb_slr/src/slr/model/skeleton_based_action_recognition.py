""" This is an implementation of the following paper:
    https://www.nature.com/articles/s41598-022-08157-5#Tab1

    It uses skeleton based action recognition, we want to check how it works for
    sign language recognition at gloss level
"""
# Python imports
import enum
from turtle import forward
from typing import List, Tuple, Optional
from matplotlib import axis
from numpy import void

# Local imports
from slr.data_processing.image_parser import PoseValues, JointData, PoseGraph
from slr.model.data_ingestor import DataIngestor

# Third party imports
import torch
import torch.nn as nn

class DataPreprocessor:
    """Convert the data we extract from videos to a format that is required for the action recognition model
    """

    def process(self, sign : List[PoseValues]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert from data inside a pose value to a torch tensor

        Args:
            data (PoseValues): pose skeleton description`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Multi representation matrix P, and velocity matrix V
        """
        # Create a matrix with the data for this sign
        P = self.create_joint_matrix(sign)

        # Create helper W matrix to compute bones representation
        W = self._create_helper_W_matrix_from_graph(sign)

        # Use W matrix to create bones representation
        bones_matrix = torch.matmul(P , W)

        I = torch.cat((P, bones_matrix))

        # Compute velocity matrix
        V = self._create_velocity_matrix_from_joint_matrix(P)

        return I, V


    def create_joint_matrix(self, sign : List[PoseValues], include_face : bool = True) -> torch.Tensor:
        """Create a joint data matrix with the data corresponding to the sign specified by a given list of poses

        Args:
            sign (List[PoseValues]): a sign defined as a list of poses

        Returns:
            torch.Tensor: A matrix C x T x V, where:
                - C is the channel size
                - T is the amount of frames (== len(sign))
                - V is the amount of joints
            Note: In the original paper, this matrix should be C x V x T, but it makes no sense with the 
            following matrix multiplications, so we use C x T x V instead.
        """
        # Get matrix size
        C = JointData.n_channels()
        V = PoseGraph.get_expected_joint_amount()
        T = len(sign)

        # Init resulting tensor
        result = torch.zeros((C,T,V))

        # Fill resulting tensor
        for (t, pose) in enumerate(sign):
            pose_graph = pose.as_graph(include_face)
            for (v, joint) in enumerate(pose_graph.joint_data):
                joint_data = [joint.x, joint.y, joint.z, joint.visibility]
                for (c, value) in enumerate(joint_data):
                    result[c,t,v] = float(value)

        return result
        

    def _create_helper_W_matrix_from_graph(self, sign : List[PoseValues]) -> torch.Tensor:
        """Create a matrix W that is used to create the bones representation for the given graph

        Args:
            sign (List[PoseGraph]): Graph whose bones will be defined

        Returns:
            torch.Tensor: A matrix such that al diagonal elements are set to 1 (an identity matrix) and if 
            there's and edge a -> b, then W[b,a] == -1
        """
        assert len(sign) > 1, "There should be at the least 1 graph in graph list to create W matrix"

        T = len(sign)
        result = torch.eye(543) #TODO should update to ignore face vertex

        # Assume that the connectivity of the first graph does not changes between graphs
        sample_graph = sign[0].as_graph(False)

        # Set up -1 in corresponding cell
        for edge in sample_graph.edges:
            result[edge.start, edge.end] = -1

        return result

    def _create_velocity_matrix_from_joint_matrix(self, P : torch.Tensor) -> torch.Tensor:
        """Generate a tensor with a velocity matrix V built from the joint data matrix C x T x V

        Args:
            P (torch.Tensor): C x T x V matrix with the data for a single sign

        Returns:
            torch.Tensor: Velocity matrix, concatenation of every joint velocity
        """

        # Init result 
        joint_V = torch.zeros(P.shape)
        (_, T, V) = P.shape

        # Compute joint velocity
        for t in range(1, T):
            joint_V[:, t] = P[:, t] - P[:, t-1]
        
        joint_V[-1] = torch.zeros((T,V))

        
        # Compute edge velocity
        bone_V = torch.zeros(P.shape)
        for t in range(1, T):
            for v in range(1,V):
                bone_V[:, t, v] = (P[:, t, v] - P[:, t, v-1]) - (P[:, t-1, v] - P[:, t-1, v-1])

        bone_V[-1] = torch.zeros((T,V))

        return torch.cat((joint_V, bone_V))

class ActionRecognitionModel(nn.Module):
    """Action recognition model based on the paper: https://www.nature.com/articles/s41598-022-08157-5#Tab1
        "An efficient self-attention network for skeleton-based action recognition"
    """

    def __init__(self, channels : int = 8, attention_block_features : int = 64) -> None:
        super().__init__()

        # Input stage
        self.velocity_convolution_1 = nn.Conv2d(channels, 1, 1)
        self.velocity_convolution_2 = nn.Conv2d(1, 1, 1)

        self.skel_convolution_1 = nn.Conv2d(channels, 1, 1)
        self.skel_convolution_2 = nn.Conv2d(1, 1, 1)

        # Self attention block 1
        # TODO test with linear models with no bias
        self.W_theta = nn.Linear(1, attention_block_features)
        self.W_void = nn.Linear(1, attention_block_features)
        self.W_g = nn.Linear(1, attention_block_features)
        self.conv_h = nn.Conv2d(attention_block_features, 1, 1)


    def forward(self, joint_matrix : torch.Tensor, vel_matrix : torch.Tensor):
        V = self.velocity_convolution_1(vel_matrix)
        V = self.velocity_convolution_2(V)
        V = torch.relu(V)

        I = self.skel_convolution_1(joint_matrix)
        I = self.skel_convolution_2(I)
        I = torch.relu(I)

        Z = V + I # 1 T V

        # TODO: tengo que añadir el bloque Z = cat(Z, J) + T, pero no se de donde salen J y T

        Z = self._self_attention_block_1(Z)


    
    def _self_attention_block_1(self, Z : torch.Tensor) -> torch.Tensor:
        # Z is C T V

        # Note: I had to transpose Z so it matches with torch linear model definition
        transposed_Z = Z.transpose(0,2)      # V T C
        theta_1 = self.W_theta(transposed_Z) # V T C
        void_1 = self.W_void(transposed_Z)   # V T C
        g_1 = self.W_g(transposed_Z)         # V T C

        (V, T, C) = theta_1.shape

        theta_1 = theta_1.reshape(V*T, C)
        void_1 = void_1.reshape(C,T*V)

        #result = torch.softmax(torch.matmul(theta_1, void_1), 0) # TV x TV
        result = torch.matmul(theta_1, void_1)# TV x TV

        g_1 = g_1.reshape(V*T, C)

        result = torch.matmul(result, g_1) # TV x C
        result = result.reshape(V, T, C)
        result = result.transpose(0,2) # C T V

        result = self.conv_h(result)

        return result + Z








    
    