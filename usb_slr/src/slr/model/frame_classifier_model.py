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

    def __init__(self, num_classes : int, image_size : int = 128) -> None:
        super().__init__()

        # Model setup
        self._num_classes = num_classes


        # Layers
        self._bn = nn.BatchNorm2d(2)

        self._cnn_seq_1 = nn.Sequential(
            nn.Conv2d(2, 10, 8, 1), nn.ReLU(inplace=True),  nn.MaxPool2d(2,2), nn.BatchNorm2d(10), nn.Dropout(0.2), # MaxPool(2,2)
            nn.Conv2d(10, 12, 5, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(12), nn.Dropout(0.2),
            nn.Conv2d(12, 14, 3, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(14), nn.Dropout(0.2), # MaxPool(2,2)
            # nn.Conv2d(14, 16, 3, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(16), nn.Dropout(0.2),
            nn.Conv2d(14, 18, 3, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2), nn.BatchNorm2d(18), nn.Dropout(0.2), # MaxPool(2,2)
        )

        

        self._fc = nn.Sequential(
            nn.Linear(18*5*5, 512), nn.Dropout(0.6), nn.Tanh(),
            nn.Linear(512, 256), nn.Dropout(0.6), nn.Tanh(),
            nn.Linear(256, num_classes), nn.Tanh()
        )
        
    def forward(self, batch_of_images : torch.Tensor) -> torch.Tensor:
        shape = batch_of_images.shape

        y = self._bn(batch_of_images)

        y = self._cnn_seq_1(y)

        y = torch.flatten(y, 1)

        y = self._fc(y)

        return torch.softmax(y, 1)
