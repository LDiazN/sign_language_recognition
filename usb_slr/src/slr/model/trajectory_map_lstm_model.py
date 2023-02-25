"""With this model we will try to classify frames for multiple 
signs. This way, we can use this pretrained model to trajectory maps in order to 
classify signs more accurate
"""

# Local imports
from collections import defaultdict
from typing_extensions import Self
from slr.data_processing.image_parser import PoseValues, PoseGraph
from slr.dataset_manager.dataset_managers import SignDescription, ArgentinaDatasetManager, MicrosoftDatasetManager, PeruDatasetManager
from slr.model.trainer import TrainResult
from slr.model.utils import Cacher
from slr.model.data_ingestor import DataIngestor, FeatureTransformer
from slr.data_processing.image_parser import PoseValues
from slr.model.trainer import Trainer


# Third party imports
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import wandb
import numpy as np
import cv2
import termcolor as c
import pandas as pd
import matplotlib.pyplot as plt


# Python imports
from random import randint
from typing import Any, List, Optional, Tuple, Dict
from torchvision.models import mobilenet_v3_small
from pathlib import Path


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

class TMLSTMCLassifier(nn.Module):
    """Trajectory Maps LSTM Classifier, use dual channel input from an LSTM layer and a trajectory map 
    to classify a sign
    """

    def __init__(self, 
        num_classes : int, 
        image_size : int = 128, 
        cnn_starting_channels : int = 10,
        cnn_channel_increase_step : int = 2,
        lstm_hidden_size : int = 32, 
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
            nn.Linear(n_frames * lstm_hidden_size, intermediate_output_size), nn.Dropout(lstm_dropout), nn.LeakyReLU(negative_slope=0.01)
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
        y_lstm = nn.LeakyReLU()(y_lstm) # (batch, keyframes, feature_len) 
        y_lstm = torch.flatten(y_lstm, start_dim=1) # (batch,  keyframes * feature_len)
        y_lstm = self._lstm_fc(y_lstm) # 
        # ---------------------------------------------

        y = self._fc(y)
        y_lstm = self._fc(y_lstm)
    
        y = (y + y_lstm) / 2 # element wise mean

        return torch.softmax(y, 1)

class TMLSTMCLassifierMobilenet(nn.Module):
    """Trajectory Maps LSTM Classifier, use dual channel input from an LSTM layer and a trajectory map 
    to classify a sign. Use mobilenet as backend for spatial features extraction
    """

    def __init__(self, 
        num_classes : int, 
        image_size : int = 128, 
        cnn_starting_channels : int = 10,
        cnn_channel_increase_step : int = 2,
        lstm_hidden_size : int = 32, 
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

        # Set up mobilenet to use a different classifier size 
        self._mobilenet = mobilenet_v3_small()
        self._mobilenet.classifier[-1] = torch.nn.Linear(1024, num_classes)
        

        intermediate_output_size = (cnn_starting_channels + 3*cnn_channel_increase_step)*5*5 # 18, 5, 5

        self._lstm_seq = nn.Sequential(
            nn.LSTM(input_size = lstm_feature_len, hidden_size = lstm_hidden_size, num_layers = lstm_num_layers, batch_first = True),
        )

        self._lstm_fc = nn.Sequential( 
            nn.Linear(n_frames * lstm_hidden_size, intermediate_output_size), nn.Dropout(lstm_dropout), nn.LeakyReLU(negative_slope=0.01)
        )

        self._fc = nn.Sequential(
            nn.Linear(intermediate_output_size, fc_intermediate_size_1), nn.Dropout(0.2), nn.Tanh(),
            nn.Linear(fc_intermediate_size_1, fc_intermediate_size_2), nn.Dropout(0.2), nn.Tanh(),
            nn.Linear(fc_intermediate_size_2, num_classes)
        )

    def forward(self, batch_of_images_and_vecs : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        batch_of_images, lstm_vecs = batch_of_images_and_vecs

        #y = self._bn(batch_of_images)

        y = self._mobilenet(batch_of_images)

        y = torch.flatten(y, 1)

        # -- < LSTM Block > ---------------------------
        y_lstm, _ = self._lstm_seq(lstm_vecs)
        y_lstm = nn.LeakyReLU()(y_lstm) # (batch, keyframes, feature_len) 
        y_lstm = torch.flatten(y_lstm, start_dim=1) # (batch,  keyframes * feature_len)
        y_lstm = self._lstm_fc(y_lstm) # 
        # ---------------------------------------------

        # y = self._fc(y)
        y_lstm = self._fc(y_lstm)
    
        y = (y + y_lstm) / 2 # element wise mean

        return torch.softmax(y, 1)

class TMLSTMClassifierTrainer:
    """
    Main class for performing the entire training workflow with this model, intended to be used 
    by the training command
    """

    VALID_DATASETS =  ["ms", "peru", "lsa64"]

    def __init__(
        self,
        dataset : str, 
        output_dir : Path, 
        profile_model : bool, 
        num_epochs : int, 
        num_folds : int, 
        cache_dir : Path, 
        num_classes : int,
        num_frames : int,
        experiment_name : str,
        use_mobilenet : bool = False,
        use_wandb : bool = True,
        test_size : float = 0.2,
        image_size  : int = 128,
        joint_radius : int = 3,
        acceptance_th : float = 0.8,
        batch_size : int = 32,
        use_kfolds : bool = True
    ):

        # Check if dataset is a valid one
        if dataset not in self.VALID_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset}. Valid options: {self.VALID_DATASETS}")

        assert 0 < test_size <= 1, "Invalid value for test_size"

        # Training loop set up
        self._dataset = dataset
        self._output_dir = output_dir
        self._profile_model = profile_model
        self._num_epochs = num_epochs
        self._num_folds = num_folds
        self._cache_dir = cache_dir
        self._num_classes = num_classes
        self._num_frames = num_frames
        self._use_mobilenet = use_mobilenet
        self._use_wandb = use_wandb
        self._test_size = test_size
        self._image_size = image_size
        self._joint_radius = joint_radius
        self._experiment_name = experiment_name
        self._acceptance_th = acceptance_th
        self._batch_size = batch_size
        self._use_kfolds = use_kfolds

        # if we want to use cuda
        self._use_cuda = torch.cuda.is_available()
        self._model = TMLSTMCLassifier(num_classes, n_frames = num_frames) if not use_mobilenet else TMLSTMCLassifierMobilenet(num_classes, n_frames = num_frames)
        self._model.to("cuda")

    def run(self):
        """
            Perform experiment workflow from start to finish
        """

        # Set up environment
        c.cprint("Preparing environment...", color='blue')
        if self._use_wandb:
            wandb.init(project="usb-slr", entity="usb-slr")

        if self._use_cuda:
            torch.cuda.set_device(torch.device("cuda:0"))
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Get data for training
        c.cprint("Reading data for training...", color='blue')
        X_train, X_val, Y_train, Y_val, labelmap = self.read_data()

        # Now convert data to dataloaders so they can be used for training
        tensor_train_x = torch.Tensor(X_train).cpu() # (amount, n_frames, n_features)
        tensor_train_y = torch.Tensor(Y_train).cpu()
        tensor_val_x = torch.Tensor(X_val).cpu()
        tensor_val_y = torch.Tensor(Y_val).cpu()

        c.cprint("Processing data for training...", color='blue')
        # Process data: Generate trajectory maps, lstm matrices. Cache it 
        # to avoid expensive computation over and over again
        args = {
            "tensor_train_x" : tensor_train_x,
            "tensor_train_y" : tensor_train_y,
            "tensor_val_x" : tensor_val_x,
            "tensor_val_y" : tensor_val_y
        }

        cache_processed = Cacher(self._process_data, cache_name= "processed_data", cache_path=str(self._cache_dir), args=args)
        tensor_train_x, tensor_train_y, tensor_val_x, tensor_val_y, lstm_tensor_train_x, lstm_tensor_val_x = cache_processed.get()

        # Now save images used for training
        c.cprint(f"Saving images in: {self.images_output_dir} ...", color='blue')
        count_per_class = self._save_images(tensor_train_x, tensor_train_y, labelmap or {})
        self._save_images(tensor_val_x, tensor_val_y, labelmap or {}, count_per_class)

        # Start training
        c.cprint("Dataset: ", color="yellow")
        c.cprint(f"\t {self._dataset}", color='blue')
        c.cprint("Model: ", color="yellow")
        c.cprint(str(self._model), color='blue')
        c.cprint("Optimizer: ", color = 'yellow')
        c.cprint(str(self.get_optimizer()), color = 'blue')
        c.cprint(f"Seed is: {torch.seed()}", color = 'yellow')
        c.cprint("Starting training", color = "green")
        train_dataset = TMLSTMDataset(tensor_train_x, tensor_train_y, lstm_data=lstm_tensor_train_x)
        val_dataset = TMLSTMDataset(tensor_val_x, tensor_val_y, lstm_data=lstm_tensor_val_x)
        results, trainer = self.run_training(train_dataset, val_dataset, self._num_classes, self._use_kfolds)

        c.cprint("Storing results...", color = "green")
        self.write_results(results, labelmap or {}, trainer)



    def read_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[int, str]]]:
        """
            Read training data from local storage
        """

        # Generate data to train depending on using cache or not
        cacher = Cacher(self._generate_data, "train_data",  cache_path=str(self._cache_dir))
        X_train, X_val, Y_train, Y_val, label_map = cacher.get()

        # Sanity check
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(Y_train))
        assert not np.any(np.isnan(X_val))
        assert not np.any(np.isnan(X_val))

        return X_train, X_val, Y_train, Y_val, label_map
        
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[int, str]]]:
        if self._dataset == "ms":
            dataset_manager = MicrosoftDatasetManager()
            filter_fn = self._filter_ms
        elif self._dataset == "peru":
            dataset_manager = PeruDatasetManager()
            filter_fn = lambda a, b: True
        elif self._dataset == "lsa64":
            dataset_manager = ArgentinaDatasetManager()
            filter_fn = self._filter_lsa64
        else:
            assert False, f"Unrecognized dataset: '{self._dataset}'"

        data_ingestor = DataIngestor(dataset_manager=dataset_manager) # type: ignore
        return data_ingestor.generate_train_data(predicate=filter_fn, padding_func=FeatureTransformer.pad_with_zeroes, normalize_location=True, reduce_labels=True, test_size=self._test_size) # type: ignore


    def _filter_ms(self, features : np.ndarray, description : SignDescription) -> bool:
        """
            Checks if the specified row of features is valid for the current training
        """
        return description.label in self._labels_ms(self._num_classes) and len(features) > 0
    
    def _filter_lsa64(self, features : np.ndarray, description : SignDescription) -> bool:
        """
            Checks if the specified row of features is valid for the current training
        """
        return description.label in self._labels_lsa64(max_size = self._num_classes) and len(features) > 0

    def _labels_lsa64(self, mandatory_labels : List[int] = [], max_size : int = 32) -> List[int]:
        """
            Labels used in argentina dataset
        """
        assert  len(mandatory_labels) <= max_size and \
                all( 1 <= l <= 64 for l in mandatory_labels) and \
                len(set(mandatory_labels)) == len(mandatory_labels) and\
                0 < max_size <= 64, \
                "Invalid mandatory labels."

        # Randomly select a new label that's not in the list of labels
        while len(mandatory_labels) < max_size:
            new_label = randint(1, 64)
            if new_label not in mandatory_labels:
                mandatory_labels.append(new_label)

        return mandatory_labels

    def _labels_ms(self, max_size : int = 20) -> List[int]:
        """
        Labels used in ms dataset
        """
        assert 0 < max_size < 1000
        allowed = [3,
                19,
                15,
                8,
                33,
                11,
                1,
                14,
                12,
                51,
                50,
                2,
                29,
                79,
                23,
                78,
                9,
                31,
                61,
                25,
                10,
                7,
                44,
                17,
                75,
                39,
                48,
                59,
                21,
                45,
                65,
                95,
                76,
                34,
                64,
                32,
                13,
                83,
                66,
                92,
                47,
                43,
                52,
                72,
                24,
                28,
                6,
                71,
                26,
                99,
                16]
        # Note that we take specifically the labels in this order since
        # not every label is the same, some have more samples than others.
        # Labels are sorted by instance amount
        return allowed[:max_size]

    @property
    def cached_data_file(self) -> Path:
        return Path(self._cache_dir, f"cache_{self._dataset}.pkl")
    
    @property
    def cached_processed_data_file(self) -> Path:
        return Path(self._cache_dir, f"cache_{self._dataset}_processed.pkl")
    
    def _process_data(self, 
        tensor_train_x : torch.Tensor, 
        tensor_train_y : torch.Tensor, 
        tensor_val_x : torch.Tensor, 
        tensor_val_y : torch.Tensor
        ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

        # Preprocess data: create pose values from arrays to create matrices
        poses_train_x = [[PoseValues.from_array(r) for r in s if r.count_nonzero() != 0] for s in tensor_train_x]
        poses_val_x = [[PoseValues.from_array(r) for r in s if r.count_nonzero() != 0] for s in tensor_val_x]

        # Filter out empty signs
        new_poses_train_x = []  
        new_poses_val_x = []
        new_train_y = []
        new_val_y = []

        for (x,y) in zip(poses_train_x, tensor_train_y):
            if not x:
                continue

            new_poses_train_x.append(x)
            new_train_y.append(y.reshape(1, *y.shape))

        poses_train_x = new_poses_train_x
        del new_poses_train_x
        tensor_train_y = torch.cat(new_train_y)
        del new_train_y


        for (x,y) in zip(poses_val_x, tensor_val_y):
            if not x:
                continue

            new_poses_val_x.append(x)
            new_val_y.append(y.reshape(1, *y.shape))

        poses_val_x = new_poses_val_x
        del new_poses_val_x
        tensor_val_y = torch.cat(new_val_y)
        del new_val_y


        processor = DataPreprocessor()

        tensor_train_x, lstm_tensor_train_x = processor.process_many(poses_train_x, joint_pos_radius=self._joint_radius, image_size_x=self._image_size, image_size_y=self._image_size, process_type="limb_colored", n_frames=self._num_frames)
        tensor_val_x, lstm_tensor_val_x = processor.process_many(poses_val_x, joint_pos_radius=self._joint_radius,  image_size_x=self._image_size, image_size_y=self._image_size, process_type="limb_colored", n_frames=self._num_frames)

        tensor_train_y = tensor_train_y.cuda()
        tensor_val_y = tensor_val_y.cuda()

        return tensor_train_x, tensor_train_y, tensor_val_x, tensor_val_y, lstm_tensor_train_x, lstm_tensor_val_x

    @property
    def images_output_dir(self) -> Path:
        return Path(self._output_dir, "trajectory_maps")

    def _save_images(self, images_x : torch.Tensor, images_y : torch.Tensor, labelmap : Dict[int, str], count_per_class : Dict[str, int] = {}) -> Dict[str, int]:
        """
            Save images in output dir. Images are expected to be in the same format as training data
        """
        # Create output dir if not exists
        images_output_dir = self.images_output_dir
        images_output_dir.mkdir(parents=True, exist_ok=True)

        output_dict = defaultdict(lambda: 0)

        for (x,y) in zip(images_x, images_y):
            x = torch.moveaxis(x, 0, -1)
            x_class_id = torch.argmax(y, dim=0).item()
            x_class = labelmap[x_class_id] # type: ignore

            if x_class not in count_per_class:
                count_per_class[x_class] = 0

            count_per_class[x_class] += 1

            x_tensor : torch.Tensor = x
            x_img = 255 * x_tensor.cpu().numpy()
            cv2.imwrite(str(Path(images_output_dir,f"{x_class}_{count_per_class[x_class]}.png")), x_img)

        return dict(output_dict)
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=0.0005, betas=(0.9,0.999), amsgrad=True, weight_decay=0.0001)

    def run_training(self, train_dataset : Dataset, val_dataset : Dataset, n_labels : int, use_kfolds : bool = True) -> Tuple[List[TrainResult], Trainer]:

        trainer = Trainer(self._model, train_dataset, val_dataset, n_classes=n_labels, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=self.get_optimizer(), experiment_name=self._experiment_name, batch_size=self._batch_size, acceptance_th=self._acceptance_th, use_wandb=self._use_wandb)

        if use_kfolds:
            return trainer.train_k_folds(
                self.get_optimizer, 
                n_epochs=self._num_epochs, 
                k_folds=self._num_folds,
                save_stats_history=True,
                profile_model=self._profile_model
                ), trainer

        return [trainer.run_train(self._num_epochs, train_title=self._experiment_name)], trainer

    def write_results(self, results : List[TrainResult], labelmap : Dict[int, str], trainer : Trainer):
        """
        Write results to output dir
        """
        results_file_path = Path(self._output_dir, "results.txt")
        with results_file_path.open("w") as f:

            table_data = {
                "Pérdida de validación" : [r.best_loss_on_validation for r in results],
                "Pérdida de entrenamiento" : [r.best_loss_on_train for r in results],
                "Precisión de validación" : [r.best_acc_on_validation for r in results],
                "Precisión de entrenamiento" : [r.best_acc_on_train for r in results],
                "Tiempo (segundos)" : [r.train_duration for r in results]
            }

            table_data = pd.DataFrame(table_data, index=[i for i in range(1, (self._num_folds if self._use_kfolds else 1) + 1)])
            table_data.index.name = "Fold" if self._use_kfolds else "Train"

            table_data.loc['Prom.'] = table_data.mean()
            print(self._latex_with_lines(table_data, float_format="%.4f"), file=f)
            
            # Annotate used classes
            print("Classes: ", file=f)
            for class_name in labelmap.values():
                print(class_name, file=f)
            
            # Model profiling 
            for (i, r) in enumerate(results):
                if r.profile_result:
                    print(f"Profiling for fold {i+1}", file = f)
                    print(r.profile_result, file = f)

        # Writing plots
        for (i, result) in enumerate(results):

            # Save loss plot
            result.loss_history.rename({"train" : "Entrenamiento", "validation" : "Validación"})
            result.loss_history.plot()
            plt.xlabel("Época")
            plt.title(f"Pérdida: Entrenamiento vs Validación ({i+1})")
            plt.savefig(str(Path(self._output_dir, f"loss_fold_{i+1}.png")))
            plt.clf()

            # Save acc plot
            result.acc_history.rename({"train" : "Entrenamiento", "validation" : "Validación"})
            result.acc_history.plot()
            plt.xlabel("Época")
            plt.title(f"Precisión: Entrenamiento vs Validación ({i+1})")
            plt.savefig(str(Path(self._output_dir, f"accuracy_fold_{i+1}.png")))
            plt.clf()    

            # Save confusion matrix
            trainer.plot_confusion_matrix(result.confusion_matrix, np.array([i for i in range(self._num_classes)]), labelmap=labelmap, file_to_save=str(Path(self._output_dir, "conf_matrix.png")))
            plt.savefig(str(Path(self._output_dir, f"conf_matrix_fold_{i+1}.png")))
            plt.clf()

    @staticmethod
    def _latex_with_lines(df : pd.DataFrame, *args, **kwargs) -> str:
        kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                        + ['r'] * df.shape[1] + [''])
        res = str(df.to_latex(*args, **kwargs))
        return res.replace('\\\\\n', '\\\\ \\hline\n')

