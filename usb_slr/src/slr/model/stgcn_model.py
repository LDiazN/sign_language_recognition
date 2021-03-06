"""A model implemented over the ST-GCN heuristic from  https://www.youtube.com/watch?v=RRMU8kJH60Q
"""
""" This is an implementation of the following paper:
    https://www.nature.com/articles/s41598-022-08157-5#Tab1

    It uses skeleton based action recognition, we want to check how it works for
    sign language recognition at gloss level
"""
# Python imports
from typing import List, Tuple, Optional
from matplotlib.pyplot import axis

# Local imports
from slr.data_processing.image_parser import PoseValues, JointData, PoseGraph

# Third party imports
import torch
import torch.nn as nn

class DataPreprocessor:
    """Convert the data we extract from videos to a format that is required for the action recognition model
    """

    def process(self, sign : List[PoseValues], include_face : bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert from data inside a pose value to a torch tensor

        Args:
            data (PoseValues): pose skeleton description

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjacency matrix, feature matrix  with features for each node, and normalized variance tensor.
            If n = len(sign), expected size for adjacency matrix is (n,n) and feature size is [n, n_nodes, 2] and normalized variance tensor
            will have shape [n_nodes, 2]
        """

        assert len(sign) > 0, "Can't compute anything on empty sign"
        graphs = [s.as_graph(include_face=include_face) for s in sign]

        # Create node features. For now, we only use x,y for features
        joint_data = torch.tensor([ [[j.x, j.y] for j in g.joint_data] for g in graphs])

        # Create adjacency matrix
        n = graphs[0].n_nodes
        result = torch.zeros((n,n))
        for edge in graphs[0].edges:
            result[edge.start, edge.end] = 1
            result[edge.end, edge.start] = 1 # Undirected graph

        # Create variance Matrix 
        vari = torch.from_numpy(PoseGraph.joint_variance(graphs)).cuda() 

        # Normalize variance with minmax
        mini, _ = vari.min(axis=0)
        maxi, _ = vari.max(axis=0)    

        vari = (1/(maxi - mini)) * (vari - mini)

        return result, joint_data, vari

    def process_many(self, signs : List[List[PoseValues]], include_face : bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process many signs at the same time 

        Parameters:
            signs = List of signs to convert into tensors
            include_face : bool = If should include face joints. Defaults to False
        Returns:
            Two tensorws with the tensors corresponding to the adjacency matrix, node features and variance tensor of every given sign concatenated
            to form a bulk
        """
        assert len(signs) > 0, "There should be at the least one sign"

        sign_tensors = [self.process(s, include_face=include_face) for s in signs]

        adj_shape = sign_tensors[0][0].shape
        features_shape = sign_tensors[0][1].shape
        variance_shape = sign_tensors[0][1].shape

        adj_result = torch.zeros((len(sign_tensors), *adj_shape))
        features_result = torch.zeros((len(sign_tensors), *features_shape))
        variance_result = torch.zeros((len(sign_tensors), *variance_shape))

        torch.cat([m.reshape((1,*m.shape)) for (m, _, _) in sign_tensors], out=adj_result)
        torch.cat([j.reshape((1,*j.shape)) for (_, j, _) in sign_tensors], out=features_result)
        torch.cat([v.reshape((1,*v.shape)) for (_, _, v) in sign_tensors], out=variance_result)
        

        return adj_result, features_result, variance_result


class GraphConvolution(nn.Module):
    """Implement a Graph convolution operation
    Implemented based on 
    https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780

    Args:
        n_nodes : int = Amount of nodes in the graph
        feature_dim : int = Dimension of the feature vector corresponding to each node.
        batch_size  : int = Size of batch
        output_size : int | None = size of the output feature vector for each node. If output_size == feature_dim, 
            keep dimensionality. Otherwise, change dimensionality. Defaults to feature_dim
    """

    def __init__(self, n_nodes : int, feature_dim : int, batch_size : int, output_size : Optional[int] = None):
        super().__init__()

        # Set up configuration variables    
        self._n_nodes = n_nodes
        self._feature_dim = feature_dim
        self._output_size = output_size or feature_dim
        self._batch_size = batch_size

        # Set up parameters
        self._convolution_parameters = torch.nn.parameter.Parameter(torch.ones((feature_dim, self._output_size)))
        self.reset_parameters()


    def forward(self, adjacency_matrix : torch.Tensor, node_features : torch.Tensor) -> torch.Tensor:
        """Run a graph convolution operation over a graph

        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix describing graph. Size: (batch_size, n_nodes, n_nodes)
            node_features (torch.Tensor): Features for each node. Size: (batch_size, n_nodes, node_feature_dim)

        Returns:
            torch.Tensor: New node features, has shape of (batch_size, n_nodes, output_size)
        """

        batch_size, _, _ = adjacency_matrix.shape

        assert adjacency_matrix.shape == (batch_size, self._n_nodes, self._n_nodes), \
                f"The provided matrix size does not match with the expected size ({batch_size}, {self._n_nodes}, {self._n_nodes})"

        assert node_features.shape == (batch_size, self._n_nodes, self._feature_dim), \
                f"The provided node features don't match expected size ({batch_size}, {self._n_nodes}, {self._feature_dim})"

        self_looped_matrix = adjacency_matrix + torch.eye(self._n_nodes)
        normalization_matrix = self._generate_normalization_matrix(self_looped_matrix)
        temp = torch.matmul(normalization_matrix, self_looped_matrix)
        temp = torch.matmul(temp, node_features)
        temp = torch.matmul(temp, self._convolution_parameters)
        
        return temp


    def _generate_normalization_matrix(self, adjacency_matrix : torch.Tensor) -> torch.Tensor:
        """Generate a normalization matrix given the graph adjacency matrix

        Args:
            adjacency_matrix (torch.Tensor): A square adjacency matrix

        Returns:
            torch.Tensor: A normalization matrix that will normalize the convolution's output
        """
        _,r,c = adjacency_matrix.shape
        assert r == c, "The provided matrix is not square, can't create normalization matrix from it"
        D = adjacency_matrix.sum(axis=2) # type: ignore
        D = torch.cat([torch.diag(r).reshape(1,len(r), len(r)) for r in D])

        return torch.inverse(D)

    def reset_parameters(self):
        """
        Code stolen from https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6
        to improve weight initialization
        """        
        import math
        torch.nn.init.kaiming_uniform_(self._convolution_parameters, a=math.sqrt(5))

class STGCNModel(nn.Module):
    """Action recognition model based on the heuristic from https://www.youtube.com/watch?v=RRMU8kJH60Q
    
    This model it's supposed to have the following structure
        Temporal block
        Spatial block
        Temporal block (optional)

        Linear blocks (classification)
    """

    def __init__(self, batch_size : int, node_feature_dim : int, n_nodes : int, n_frames : int, n_classes : int, convolution_out_size : int = 8) -> None:
        super().__init__()

        # Network config
        self._batch_size = batch_size
        self._node_feature_dim = node_feature_dim
        self._n_nodes = n_nodes
        self._n_frames = n_frames
        self._n_classes = n_classes
        self._convolution_out_size = convolution_out_size

        # Config parameters:
        #self._temporal_block = torch.nn.LSTM(node_feature_dim * n_nodes, n_nodes * node_feature_dim, batch_first=True, num_layers=16) # ???
        self._temporal_block = torch.nn.MultiheadAttention(2 * n_nodes, num_heads=1, batch_first=True, kdim=2*n_nodes, vdim=2*n_nodes) # ???
        self._spatial_block = GraphConvolution(n_nodes, feature_dim=node_feature_dim, batch_size=batch_size, output_size=convolution_out_size)
        self._linear1 = torch.nn.Linear(convolution_out_size * n_nodes , 64)
        self._linear2 = torch.nn.Linear(64, n_classes)
        self._batch_norm = torch.nn.BatchNorm2d(node_feature_dim)
   

    def forward(self, matrices_and_features_and_variance : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adjacency_matrices, feature_matrix, variance_matrix = matrices_and_features_and_variance
        
        batch_size, _, _ = adjacency_matrices.shape

        # Batch normalization layer
        feature_matrix = feature_matrix.transpose(1,3)
        feature_matrix = self._batch_norm(feature_matrix)
        feature_matrix = feature_matrix.transpose(1,3)

        # Consistency check
        expected_adjacency_matrix_size = (batch_size, self._n_nodes, self._n_nodes)
        assert adjacency_matrices.shape == expected_adjacency_matrix_size, \
        f"Unmatching adjacency matrix size. Expected size: {(batch_size, self._n_nodes, self._n_nodes)}"

        expected_feature_matrix_size = (batch_size, self._n_frames, self._n_nodes, self._node_feature_dim)
        assert feature_matrix.shape == expected_feature_matrix_size, f"Unmatching node feature matrix size. Expected size {expected_feature_matrix_size}"

        # model evaluation 
        #   Eval attention layer
        key = feature_matrix.reshape((batch_size, self._n_frames, 2 * self._n_nodes))
        query=variance_matrix.reshape((batch_size, 1, 2 * self._n_nodes))
        y,_ = self._temporal_block(
            query, # Query
            key, # key
            key, # value
            need_weights = False
            )
        y = y.reshape(batch_size, self._n_nodes, self._node_feature_dim)

        y = self._spatial_block(adjacency_matrices, y)

        y = torch.flatten(y, start_dim=1)
        y = self._linear1(y)
        y = self._linear2(y)

        return torch.softmax(y, 1)

        pass


    def _old_forward(self, matrices_and_features_and_variance : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run the model to try to get a prediction

        Args:
            adjacency_matrices (torch.Tensor): Adjacency matrix for each graph. Expected size: (batch_size, n_nodes, n_nodes)
            feature_matrix (torch.Tensor): feature for each node in each given matrix. Expected size: (batch_size, n_frames, n_nodes, node_feature_dim)
        """

        adjacency_matrices, feature_matrix, variance_matrix = matrices_and_features_and_variance
        
        batch_size, _, _ = adjacency_matrices.shape

        # Batch normalization layer
        feature_matrix = feature_matrix.transpose(1,3)
        feature_matrix = self._batch_norm(feature_matrix)
        feature_matrix = feature_matrix.transpose(1,3)

        # Consistency check
        expected_adjacency_matrix_size = (batch_size, self._n_nodes, self._n_nodes)
        assert adjacency_matrices.shape == expected_adjacency_matrix_size, \
        f"Unmatching adjacency matrix size. Expected size: {(batch_size, self._n_nodes, self._n_nodes)}"

        expected_feature_matrix_size = (batch_size, self._n_frames, self._n_nodes, self._node_feature_dim)
        assert feature_matrix.shape == expected_feature_matrix_size, f"Unmatching node feature matrix size. Expected size {expected_feature_matrix_size}"

        # Model evaluation

        # Eval lstm model: Each graph feature list is turned into a single vector, and therefore the sequence of graphs is turned into a 
        # a sequence of vectors, which is analyzed by the temporal component
        new_feats = feature_matrix.reshape((batch_size, self._n_frames, self._n_nodes * self._node_feature_dim))
        (new_feats, _) = self._temporal_block(new_feats)
        new_feats = torch.relu(new_feats)
        new_feats = new_feats.reshape((batch_size, self._n_frames, self._n_nodes, self._node_feature_dim))


        # Run a gcn model over 
        new_rows = torch.zeros((batch_size, self._n_frames, self._n_nodes, self._convolution_out_size))
        for (i, (sign, mat)) in enumerate(zip(new_feats, adjacency_matrices)):
            new_rows[i] = self._spatial_block(mat.repeat((self._n_frames, 1,1)), sign)
            
        # Classification head
        new_feats = torch.flatten(new_rows, start_dim=1)
        new_feats = self._linear1(new_feats)
        new_feats = self._linear2(new_feats)

        return torch.softmax(new_feats, 1)

        







    
    