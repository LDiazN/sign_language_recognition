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
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_small
# Python imports
from typing import Any, List, Optional, Tuple
from torch.utils.data import DataLoader


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
                    joint_color=(1.0, 0.0,0.0),
                    #intensity_fn= lambda _: 1
                )[..., 0]

        img = torch.from_numpy(
                img
            ).cuda()

        # Create variance graph
        variance = PoseGraph.joint_variance(graphs).mean(axis=1)
        variance_col = numpy.c_[variance, numpy.zeros((len(variance), 2))]
        variance_col = numpy.repeat(variance_col[numpy.newaxis, :, :], len(graphs), axis=0)
        variance_img = PoseGraph.as_trajectory_cv_img(
            graphs, 
            width=image_size_x, 
            height=image_size_y, 
            joint_radius_px=joint_pos_radius, 
            joint_color_array=variance_col
            )[:,:,0]
        variance_img = torch.from_numpy(variance_img)

        # Create resulting image
        result = torch.zeros((image_size_x, image_size_y, 2))
        result[:,:,0] = img
        #result[:,:,1:3] = velocity[:,:,:2]
        result[:,:,1] = variance_img

        return result 

    @staticmethod
    def process_limb_colored(sign : List[PoseValues], include_face : bool = False, image_size_x : int = 256, image_size_y : int = 256, joint_pos_radius = 2) -> torch.Tensor:
        """Process dataset so that the resulting training data will be trajectory maps images colored by limb

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
        img = PoseGraph.as_trajectory_cv_img_limb_colored(
                    graphs, 
                    width=image_size_x, 
                    height=image_size_y, 
                    joint_radius_px=joint_pos_radius,
                    #intensity_fn= lambda _: 1
                )

        img = torch.from_numpy(
                img
            ).cuda()

        return img 


    def get_velocity_map(self, graphs : PoseGraph, joint_pos_radius : int, width : int, height : int) -> torch.Tensor:

        # Set up velocity map
        velocity = PoseGraph.as_velocity_map(
            graphs,
            width=width,
            height=height,
            joint_radius_px=joint_pos_radius,
        )

        velocity = torch.from_numpy(velocity).cuda()

        return velocity

    def get_variance_graph(self, graphs : PoseGraph, joint_pos_radius : int, width : int, height : int) -> torch.Tensor:

        # Set up variance map
        variance = PoseGraph.variance_graph(
            graphs,
            show_edges=False, 
            joint_color=(1,0,0), 
            joint_radius_px=joint_pos_radius,
            width=width,
            height=height)[...,0]
        variance = torch.from_numpy(variance).cuda()
        return variance

    def process_lstm_data(self, sign : List[PoseValues], include_face : bool = False) -> torch.Tensor:
        """Process a sign to return a tensor with the lstm representation of this sign (a list of tensor of size `seq_len`)

        Args:
            sign (List[PoseValues]): List of poses as numbers arrays defining a sign 
            include_face (bool, optional): If should include face information. Defaults to False.

        Returns:
            torch.Tensor: A tensor with the shape (n_frames, pose_size (more likely 75))
        """
        tensors = [torch.from_numpy(frame.concatenated(exclude_face=not include_face)).reshape((1,-1)) for frame in sign]
        tensors = torch.cat(tensors).cuda()
        tensors = tensors.type(torch.float32)
        return tensors

    def process_many(self, 
        signs : List[List[PoseValues]], 
        include_face : bool = False, 
        image_size_x : int = 256, 
        image_size_y : int = 256, 
        joint_pos_radius = 2, 
        process_type : str = "default",
        n_frames : int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process many signs into a single torch tensor

        Args:
            signs (List[List[PoseValues]]): Signs to be processed and returned into a single tensor
            include_face (bool, optional): If should include face joints. Defaults to False.
            image_size_x (int, optional): width of generated image. Defaults to 256.
            image_size_y (int, optional): height of generated image. Defaults to 256.
            joint_pos_radius (int, optional): Radius in pixels of joint positions in image. Defaults to 2.
            process_type (str): Type of process to use. Choices are:   
                - default: standard process with simple trajectory images
                - limb_colored: colored by limb (3 channel image)
            n_frames  (int): Amount of frames to cap. If less than this, remaining lstm rows will be paded with zeros

        Returns:
            (torch.Tensor, torch.Tensor): trajectory images stacked into a tensor, and lstm sequences for each sign type
        """

        assert len(signs) > 0, "Expected at the least one sign"

        # Process lstm data
        lstm_data = torch.tensor([]).cuda()
        lstm_tensors = (self.process_lstm_data(sign) for sign in signs)
        for tensor in lstm_tensors:
            shape = tensor.shape
            frames, word_len = shape
            if n_frames > frames:
                tensor = torch.cat([tensor, torch.zeros((n_frames - frames, word_len))])
                shape = tensor.shape
            elif n_frames < frames:
                tensor = tensor[:n_frames]
                shape = tensor.shape
            # Pad remainind space with zeros
            lstm_data = torch.cat([lstm_data, tensor.reshape((1, *shape))])

        # Convert signs into images
        if process_type == "default":
            process_fn = self.process
        elif process_type == "limb_colored":
            process_fn = self.process_limb_colored
        else:
            assert False, f"Invalid type of process: {process_type}"

        sign_tensors = [
                        process_fn(
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

        images_result = torch.moveaxis(images_result, -1, 1)


        return images_result, lstm_data

class TMLSTMDataset(Dataset):
    """Dataset for Trajectory map-LSTM Model
    """
    def __init__(self, images : torch.Tensor, labels : torch.Tensor, transform = None, lstm_data : Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self._images = images
        self._labels = labels
        self.transform = transform
        self.lstm_data = lstm_data

    def __getitem__(self, index: Any):

        if self.transform:
            sample_images = self.transform(self._images[index]).cuda()
        else:
            sample_images = self._images[index]

        if self.lstm_data is not None:
            return (sample_images, self.lstm_data[index]), self._labels[index]

        return sample_images, self._labels[index]

    def __len__(self):
        return len(self._labels)


