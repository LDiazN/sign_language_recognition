"""
    Initial testing version of the model
"""
# Python imports
from typing import List

# Third party imports 
import torch.nn as nn
import torch

# Local imports
from slr.data_processing.image_parser import PoseValues, PoseGraph

class DataPreprocessor:

    def process(self, sign : List[PoseValues], include_face : bool = False) -> torch.Tensor:
        sign_graphs = [pv.as_graph(include_face) for pv in sign]
        tensor_variance = torch.from_numpy(PoseGraph.joint_variance(sign_graphs)).cuda()
        sign_arrays = (tensor_variance * torch.from_numpy(g.joints_array(just_xy=True)).cuda() for g in sign_graphs)
        sign_arrays = [a.reshape(1, *a.shape) for a in sign_arrays]

        return torch.cat(sign_arrays)
    
    def process_many(self, signs : List[List[PoseValues]], include_face : bool = False) -> torch.Tensor:

        signs_tensors = (self.process(s) for s in signs)
        signs_tensors = [t.reshape(1, *t.shape) for t in signs_tensors]

        return torch.cat(signs_tensors)


class LSTMBasedModel(torch.nn.Module):
    def __init__(self, input_vector_size : int, num_classes : int) -> None:
        super().__init__()

        self._input_vector_size = input_vector_size
        self._num_classes = num_classes

        # self.lstm_0 = torch.nn.LSTM(input_size=(N_FEATURES), hidden_size=80,num_layers=1, batch_first = True)
        # self.attention = torch.nn.MultiheadAttention(80, 1, batch_first = True, kdim=N_FEATURES, vdim=N_FEATURES)
        self.lstm_1 = torch.nn.LSTM(input_size=(input_vector_size), hidden_size=64,num_layers=1, batch_first = True)
        self.lstm_2 = torch.nn.LSTM(input_size=(64), hidden_size=128, batch_first = True)
        self.lstm_3 = torch.nn.LSTM(input_size=(128), hidden_size=64, batch_first = True)
        self.linear1 = torch.nn.Linear(64, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, num_classes)

    def forward(self, x):

        batch_size, sign_len, n_joints, joint_size = x.shape
        x = x.reshape((batch_size, sign_len, n_joints * joint_size)) 
        (x, _) = self.lstm_1(x)
        x = torch.tanh(x)

        (x, _) = self.lstm_2(x)
        x = torch.tanh(x)

        (x, _) = self.lstm_3(x)
        x = torch.tanh(x)
        x = x[:, -1, :]# return sequences = False, take the last value for each sequence in the batch

        x = self.linear1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = torch.tanh(x)

        y = self.linear3(x)
        y = torch.softmax(y, 1)

        return y
