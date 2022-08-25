"""With this model we will try to classify frames for multiple 
signs. This way, we can use this pretrained model to trajectory maps in order to 
classify signs more accurate
"""

# Local imports
import numpy
from slr.data_processing.image_parser import PoseValues, PoseGraph
from slr.model.utils import compute_output_shape

# Third party imports
import torch.nn as nn
import torch


# Python imports
from typing import List, Tuple


class DataPreprocessor:
    def process(self, sign : List[PoseValues], label : torch.Tensor, include_face : bool = False, image_size_x : int = 256, image_size_y : int = 256, joint_pos_radius = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sign as a sequence os pose values to get trajectory images

        Args:
            sign (List[PoseValues]): Sign as a sequence of poses
            include_face (bool, optional): If should include face joints. Defaults to False.
            image_size_x (int, optional): width of generated image. Defaults to 256.
            image_size_y (int, optional): height of generated image. Defaults to 256.
            joint_pos_radius (int, optional): Radius in pixels of joint positions in image. Defaults to 2.

        Returns:
            torch.Tensor: Trajectory map for this sign
        """
        graphs = [s.as_graph(include_face) for s in sign]
        imgs = PoseGraph.sign_to_imgs(graphs, width=image_size_x, height=image_size_y, joint_radius_px=joint_pos_radius)
        imgs = [img[:,:,0] for img in imgs]
        imgs = torch.from_numpy(numpy.array(imgs))
        labels = label.repeat((len(sign), 1))

        return imgs, labels


    def process_many(self, signs : numpy.ndarray, labels : numpy.ndarray, include_face : bool = False, image_size_x : int = 256, image_size_y : int = 256, joint_pos_radius = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process many signs into a single torch tensor

        Args:
            signs (List[List[PoseValues]]): Signs to be processed and returned into a single tensor
            include_face (bool, optional): If should include face joints. Defaults to False.
            image_size_x (int, optional): width of generated image. Defaults to 256.
            image_size_y (int, optional): height of generated image. Defaults to 256.
            joint_pos_radius (int, optional): Radius in pixels of joint positions in image. Defaults to 2.

        Returns:
            torch.Tensor: trajectory images stacked into a tensor
        """
        # signs shape: (n_signs, n_frames, n_values)
        # We have to convert signs to a list of len n_signs with a list of pose values as elements
        pose_values = [[PoseValues.from_array(row) for row in sign if torch.count_nonzero(row) != 0] for sign in signs]

        # Store results in this vector
        result_X = torch.Tensor([]).cpu()
        result_Y = torch.Tensor([]).cpu()
        for (sign, label) in zip(pose_values, labels):
            (frames, repeated_label) = self.process(sign, label, include_face=include_face, image_size_x=image_size_x, image_size_y=image_size_y, joint_pos_radius=joint_pos_radius)
            result_X = torch.cat([result_X, frames])
            result_Y = torch.cat([result_Y, repeated_label])

        n_images, h, w = result_X.shape
        result_X = result_X.reshape((n_images, 1, h, w))
        result_X = result_X.float()
        result_Y = result_Y.float()

        return result_X, result_Y

class FrameClassifier(nn.Module):
    """Model to classify frames to gestures
    """

    def __init__(self, 
        num_classes : int, 
        image_size : int = 128, 
        cnn_starting_channels : int = 10,
        cnn_channel_increase_step : int = 2,
        lstm_hidden_size : int = 64, 
        lstm_num_layers : int = 2, 
        n_frames : int = 80,
        lstm_dropout : float = 0.6,
        lstm_feature_len : int = 258,
        fc_intermediate_size_1 : int = 256,
        fc_intermediate_size_2 : int = 128,
        ):
        super().__init__()

        assert 0 < lstm_dropout < 1, "Dropout should be un range (0,1)"

        # Model setup
        self._num_classes = num_classes


        # Layers
        #self._bn = nn.BatchNorm2d(3)

        self._cnn_seq_1 = nn.Sequential(
            nn.Conv2d(3, cnn_starting_channels, 8, 1), nn.ReLU(inplace=True),  nn.MaxPool2d(2,2), nn.BatchNorm2d(cnn_starting_channels), nn.Dropout(0.2), # MaxPool(2,2)
            nn.Conv2d(cnn_starting_channels, cnn_starting_channels, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cnn_starting_channels, cnn_starting_channels + cnn_channel_increase_step, 5, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(cnn_starting_channels + cnn_channel_increase_step), nn.Dropout(0.2),
            nn.Conv2d(cnn_starting_channels + cnn_channel_increase_step, cnn_starting_channels + 2*cnn_channel_increase_step, 3, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(cnn_starting_channels + 2*cnn_channel_increase_step), nn.Dropout(0.2), # MaxPool(2,2)
            # nn.Conv2d(14, 16, 3, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(16), nn.Dropout(0.2),
            nn.Conv2d(cnn_starting_channels + 2*cnn_channel_increase_step, cnn_starting_channels + 3*cnn_channel_increase_step, 3, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(cnn_starting_channels + 3*cnn_channel_increase_step), nn.Dropout(0.2), # MaxPool(2,2)
        )

        intermediate_output_size = (cnn_starting_channels + 3*cnn_channel_increase_step)*5*5 # 18, 5, 5

        self._lstm_seq = nn.Sequential(
            nn.LSTM(input_size = lstm_feature_len, hidden_size = lstm_hidden_size, num_layers = lstm_num_layers, batch_first = True),
        )

        self._lstm_fc = nn.Sequential( 
            nn.Linear(n_frames * lstm_hidden_size, intermediate_output_size), nn.Dropout(lstm_dropout), nn.Tanh()
        )

        self._fc = nn.Sequential(
            nn.Linear(intermediate_output_size, fc_intermediate_size_1), nn.Dropout(0.2), nn.Tanh(),
            nn.Linear(fc_intermediate_size_1, fc_intermediate_size_2), nn.Dropout(0.2), nn.Tanh(),
            nn.Linear(fc_intermediate_size_2, num_classes)
        )

    def forward(self, batch_of_images_and_vecs : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        batch_of_images, lstm_vecs = batch_of_images_and_vecs

        #y = self._bn(batch_of_images)

        y = self._cnn_seq_1(batch_of_images)

        y = torch.flatten(y, 1)

        # -- < LSTM Block > ---------------------------
        y_lstm, _ = self._lstm_seq(lstm_vecs)
        y_lstm = torch.tanh(y_lstm) # (batch, keyframes, feature_len) 
        y_lstm = torch.flatten(y_lstm, start_dim=1) # (batch,  keyframes * feature_len)
        y_lstm = self._lstm_fc(y_lstm) # 
        # ---------------------------------------------

        y = self._fc(y)
        y_lstm = self._fc(y_lstm)
    
        y = (y + y_lstm) / 2 # element wise mean

        return torch.softmax(y, 1)
