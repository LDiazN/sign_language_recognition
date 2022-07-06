"""
This model is based on the following paper:
    https://www.hindawi.com/journals/mpe/2021/6650632/
    "3D Skeletal Human Action Recognition Using a CNN Fusion Model"
    It will use skeletal data encoded into RGB images to use traditional CNN approaches
"""

# Third party imports
import cv2
from rsa import sign
import torch
import torch.nn as nn

# Python imports
from typing import Tuple, List

# Local imports 
from slr.data_processing.image_parser import PoseValues, PoseGraph


# STEP ONE: We need to create a data preprocessor capable of creating trajectory images as input for the model

class DataPreprocessor:
    def process(self, sign : List[PoseValues], include_face : bool = False, image_size_x : int = 256, image_size_y : int = 256, joint_pos_radius = 2) -> torch.Tensor:
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

        # Convert sign into graphs
        graphs = [p.as_graph(include_face=include_face) for p in sign]

        # Create trajectory map
        img = PoseGraph.as_trajectory_cv_img(
                    graphs, 
                    width=image_size_x, 
                    height=image_size_y, 
                    joint_radius_px=joint_pos_radius,
                    joint_color=(1.0, 0.0,0.0)
                )[..., 0]

        t = torch.from_numpy(
                img
            ).cuda()

        return t

    def process_many(self, signs : List[List[PoseValues]], include_face : bool = False, image_size_x : int = 256, image_size_y : int = 256, joint_pos_radius = 2) -> torch.Tensor:
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

        assert signs, "Expected at the least one sign"

        # Convert signs into images
        sign_tensors = [
                        self.process(
                            s, 
                            include_face=include_face, 
                            image_size_x=image_size_x, 
                            image_size_y=image_size_y, 
                            joint_pos_radius=joint_pos_radius
                        ) 
                        for s in signs]
        
        # Buffer to stack images into
        images_result = torch.zeros((len(sign_tensors), *sign_tensors[0].shape))

        # Flatten list of images into a tensor
        torch.cat([img.reshape((1, *img.shape)) for img in sign_tensors], out=images_result)

        return images_result


# STEP TWO: We need to create the dataset to load data during training

class TMDataset:
    pass

# STEP THREE: Create model itself

class TMCNNModel(nn.Module): # Trajectory Map CNN Model
    """
        A CNN model based on clasifying trajectory maps generated from skeletal graphs    
    """

    def __init__(self, num_classes : int) -> None:
        super().__init__()

        self._num_classes = num_classes

        self._cnnf = nn.Conv2d(1, 4, 4)
        self._spatial_pooling = nn.MaxPool2d(4) # Test with more types of pooling
        self._cnnr = nn.Conv2d(4, 4, 4)
        self._linear1 = nn.Linear(4*60*60, 64   )
        self._linear2 = nn.Linear(64    , num_classes)


    def forward(self, batch_of_images : torch.Tensor) -> torch.Tensor:
        
        batch_size, w, h = batch_of_images.shape
        batch_of_images = batch_of_images.reshape((batch_size, 1, w, h))

        y = self._cnnf(batch_of_images)
        y = self._spatial_pooling(y)
        y = self._cnnr(y)
        y = torch.flatten(y, 1)
        y = self._linear1(y)
        y = self._linear2(y)

        return torch.softmax(y, 1)