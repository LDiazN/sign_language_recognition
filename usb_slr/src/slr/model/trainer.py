"""
    This module will help you to perform a training with a provided model and the required data.
    You should prepare your training data and dataloaders
"""

# Third party imports
from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchmetrics import Accuracy

# Python imports
from datetime import datetime

# local imports


class Trainer:
    """
        The trainer class will perform training for a model with the specified 
        data as input.
    """

    def __init__(self, model : nn.Module, train_data : DataLoader, valid_data : DataLoader, loss_fn : nn.CrossEntropyLoss = nn.CrossEntropyLoss(), optimizer : Optional[torch.optim.Optimizer] = None):
        self._model = model
        self._train_data = train_data
        self._valid_data = valid_data
        self._loss_fn = loss_fn
        self._optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9,0.999), amsgrad=False)

    def train_one_epoch(self, epoch_index : int, acceptance_th : float = 0.5, tb_writer : Optional[SummaryWriter] = None ) -> Tuple[float, float]:
        """  Perform training for a single epoch, and return loss and accuracy 

        Args:
            epoch_index (int): index of this epoch
            acceptance_th (float): value in [0,1] that represents the minimum confidence required to consider a category as selected. 
                                    For example, if = 0.5, the category in the softmax output of the classification model will consider any class > 0.5 as the predicted class
            tb_writer (Optional[SummaryWriter]): object to write to tensorboard logs

        Returns:
            Tuple[float, float]: (Accuracy, loss) for this epoch
        """

        assert 0 <= acceptance_th <= 1

        loss_gathered = 0
        train_data = self._train_data
        model = self._model
        loss_fn = self._loss_fn
        optim = self._optimizer

        n_batches = 0 # Use element count to compute avg loss and accuracy

        accuracy_counter = Accuracy()

        for i, data in enumerate(train_data):
            inputs, labels = data
            n_batches += 1

            # eval inputs for each batch
            optim.zero_grad()
            outputs = model(inputs)

            # Compute loss and optimize
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()

            accuracy_counter.update(outputs, labels.argmax(1))

            # Collect metrics 
            loss_gathered += loss.item()

            # Log to torch logs if a writer is provided
            if tb_writer and i % 3 == 0:
                loss_so_far = loss_gathered / n_batches
                acc_so_far = accuracy_counter.compute()
                tb_x = epoch_index * len(train_data) + i + 1
                tb_writer.add_scalar('Loss/train', loss_so_far, tb_x) # Write loss in train to tensorboard logs
                tb_writer.add_scalar('Acc/train', acc_so_far, tb_x) # Write loss in train to tensorboard logs


        accuracy = accuracy_counter.compute().item()
        loss_gathered /=  n_batches

        return (accuracy, loss_gathered)

    def run_train(self, n_epochs : int, acceptance_th : float = 0.5):
        """Run a training process for the specified amount of epochs

        Args:
            n_epochs (int): Amount of epochs to train
            acceptance_th (float) : value in [0,1] that represents the minimum confidence required to consider a category as selected. 
                                    For example, if = 0.5, the category in the softmax output of the classification model will consider 
                                    any class > 0.5 as the predicted class
        """

        model = self._model
        loss_fn = self._loss_fn

        valid_data = self._valid_data

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/sign_lang_recognition{}'.format(timestamp))

        # Keep track of best metrics
        best_loss = float('inf')
        best_acc = -1.

        accuracy_counter = Accuracy()

        # Main training loop
        iter = tqdm.tqdm(range(n_epochs))
        for i in iter:

            # Run a single train step
            model.train(True)
            avg_acc, avg_loss = self.train_one_epoch(i, tb_writer=writer, acceptance_th=acceptance_th)
            model.train(False)


            # Compute validation metrics
            n_batches = 0
            valid_acc = 0
            valid_loss = 0
            for (inputs, labels) in valid_data:

                outputs = model(inputs)

                accuracy_counter.update(outputs, labels.argmax(1))

                valid_loss += loss_fn(outputs, labels).item()

                n_batches += 1

            valid_acc = accuracy_counter.compute().item()
            valid_loss /= n_batches 

            iter.set_description(f"[Epoch: {i+1} / {n_epochs}] Train Acc: {round(avg_acc,4)} | Train Loss: {round(avg_loss,4)} | Val Acc: {round(valid_acc,4)} | Val Loss: {round(avg_loss,4)}")

            # Log to tensorboard
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : valid_loss },
                    i + 1)
            
            writer.add_scalars('Training vs. Validation Accuracy',
                    { 'Training' : avg_acc, 'Validation' : valid_acc },
                    i + 1)
            writer.flush()

            # Keep track of best model
            best_acc = max(valid_acc, best_acc)
            best_loss = min(best_loss, valid_loss)
            # TODO save to disk the best model

        print(f"Training finished, best loss on validation: {best_loss}, best accuracy on validation {best_acc}")



    def _get_success_count(self, labels : torch.Tensor, preds : torch.Tensor, acceptance_th : float) -> int:
        """Get ammount of correct answers in the pred tensor according to the labels tensor and the acceptance treshold

        Args:
            labels (torch.Tensor): Vector of size [n, l] where l is the ammount of labels, and there's only one 1 per row
            preds (torch.Tensor): Predictions Vector of size [n, l] where every row is softmaxed
            acceptance_th (float): minimum value to accept a class as selected by the model

        Returns:
            int: how many correct answers
        """
        assert labels.shape == preds.shape, "incompatible tensor shapes"

        pred_labels = (preds > acceptance_th).float().argmax(1)
        actual_labels = labels.argmax(1)

        return int((pred_labels == actual_labels).float().sum().item())