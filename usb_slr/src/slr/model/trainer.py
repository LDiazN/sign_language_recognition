"""
    This module will help you to perform a training with a provided model and the required data.
    You should prepare your training data and dataloaders
"""

# Third party imports
import dataclasses
from typing import Any, Callable, Dict, List, Tuple, Optional, Type
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchmetrics import Accuracy
from sklearn.model_selection import KFold
from stringcolor import cs
import itertools
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.optim.lr_scheduler import StepLR,_LRScheduler

# Python imports
from dataclasses import dataclass, asdict
from datetime import datetime

# local imports



@dataclass
class TrainResult:
    """Result after a training process
    """

    best_loss_on_validation : float
    best_acc_on_validation : float
    best_loss_on_train : float
    best_acc_on_train : float
    n_epochs : int
    n_classes : int
    train_duration : float 

    def as_report(self) -> str:
        """Return a human-readable version of this data

        Returns:
            str: Human readable version, might contain colored strings
        """

        train_color = 'cyan'
        validation_color = 'gold2'
        loss_color = 'red3'
        acc_color = "steelblue"

        train_str = str(cs("train", train_color))
        valid_str = str(cs("validation", validation_color))
        loss_str = str(cs("loss", loss_color))
        acc_str = str(cs("accuracy", acc_color))

        result = "🏋️  -- < Training Result > ----------------  🏋️\n"
        result += f"Epochs: {self.n_epochs} | Classes: {self.n_classes}\n"
        result += f"\t- Best {loss_str} on {valid_str}: {str(cs(self.best_loss_on_validation, 'blue'))}\n"
        result += f"\t- Best {acc_str} on {valid_str}: {TrainResult._get_color_for_acc(self.best_acc_on_validation)}\n"
        result += f"\t- Best {loss_str} on {train_str}: {str(cs(self.best_loss_on_train, 'blue'))}\n"
        result += f"\t- Best {acc_str} on {train_str}: {TrainResult._get_color_for_acc(self.best_acc_on_train)}\n"
        result += f"Training time: {cs(str(round(self.train_duration, 3)), 'purple')}\n"
    
        return result

    @staticmethod
    def _get_color_for_acc(acc : float) -> str:
        """Get color for the specified accuracy, the greener the color, the better the accuracy

        Args:
            acc (float): Accuracy to score

        Returns:
            _type_: _description_
        """
        val = round(acc, 4)
        if 0 <= acc < 0.25:
            color = "red"
        elif .25 <= acc < .50:
            color = "orange"
        elif .50 <= acc < .75:
            color = "yellow"
        elif .75 <= acc <= 1:
            color = "green"
        else:
            raise ValueError(f"Invalid value for accuracy: {val}")

        return str(cs(str(val), color))
class Trainer:
    """
        The trainer class will perform training for a model with the specified 
        data as input.
    """

    def __init__(self, model : nn.Module, train_data : Dataset, valid_data : Dataset, loss_fn : nn.CrossEntropyLoss = nn.CrossEntropyLoss(), optimizer : Optional[torch.optim.Optimizer] = None, experiment_name : str = "sign_lang_recognition", batch_size : int = 10, acceptance_th = 0.5, use_wandb : bool = False, lr_scheduler : Optional[_LRScheduler] = None):
        self._model = model
        self._train_data = train_data
        self._valid_data = valid_data
        self._loss_fn = loss_fn
        self._optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9,0.999), amsgrad=False)
        self._experiment_name = experiment_name
        self._batch_size = batch_size
        self._acceptance_th = acceptance_th
        self._use_wandb = use_wandb
        self._lr_scheduler = lr_scheduler


    def train_one_epoch(self, 
                        epoch_index : int, 
                        acceptance_th : float = 0.5, 
                        tb_writer : Optional[SummaryWriter] = None, 
                        train_data : Optional[DataLoader] = None,
                        optim : Optional[torch.optim.Optimizer] = None,
                        model : Optional[nn.Module] = None
                         ) -> Tuple[float, float]:
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
        train_data = train_data or DataLoader(self._train_data, batch_size=self._batch_size)
        model = model or self._model
        loss_fn = self._loss_fn
        optim = optim or self._optimizer

        n_batches = 0 # Use element count to compute avg loss and accuracy

        accuracy_counter = Accuracy(acceptance_th)

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
                loss_so_far = float(loss_gathered / n_batches)
                acc_so_far = accuracy_counter.compute()
                tb_x = epoch_index * len(train_data) + i + 1
                tb_writer.add_scalar('Loss/train', loss_so_far, tb_x) # Write loss in train to tensorboard logs
                tb_writer.add_scalar('Acc/train', acc_so_far, tb_x) # Write loss in train to tensorboard logs


        accuracy = accuracy_counter.compute().item()
        loss_gathered /=  n_batches

        return (accuracy, loss_gathered)

    def run_train(
        self, 
        n_epochs : int, 
        acceptance_th : float = 0.5, 
        train_dataloader : Optional[DataLoader] = None, 
        valid_dataloader : Optional[DataLoader] = None, 
        optim : Optional[torch.optim.Optimizer] = None,
        model : Optional[nn.Module] = None,
        train_title : Optional[str] = None
        ) -> TrainResult:
        """Run a training process for the specified amount of epochs

        Args:
            n_epochs (int): Amount of epochs to train
            acceptance_th (float) : value in [0,1] that represents the minimum confidence required to consider a category as selected. 
                                    For example, if = 0.5, the category in the softmax output of the classification model will consider 
                                    any class > 0.5 as the predicted class
        """

        model = model or self._model
        loss_fn = self._loss_fn

        # Set up wandb monitoring in case is needed
        if self._use_wandb:
            wandb.watch(model, criterion=loss_fn, log="all")
            configs = {
                "n_epochs" : n_epochs,
                "acceptance_th" : acceptance_th
            }

            if train_title:
                configs["title"] = train_title

            wandb.config = configs


        valid_data = valid_dataloader or DataLoader(self._valid_data, batch_size=self._batch_size)
        train_dataloader = train_dataloader or DataLoader(self._train_data, batch_size=self._batch_size,)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/{}{}'.format(self._experiment_name, timestamp))

        # Keep track of best metrics
        best_loss = float('inf')
        best_acc = -1.

        best_train_loss = float('inf')
        best_train_acc = -1.

        accuracy_counter = Accuracy(threshold=acceptance_th)

        # Main training loop
        iter = tqdm.tqdm(range(n_epochs))
        timer = ScopedTimer("Trainning")
        for i in iter:

            # Run a single train step
            model.train(True)
            avg_acc, avg_loss = self.train_one_epoch(i, tb_writer=writer, acceptance_th=acceptance_th, optim=optim, model=model, train_data=train_dataloader)
            if self._lr_scheduler:
                self._lr_scheduler.step()
            model.train(False)

            best_train_acc = max(best_train_acc, avg_acc)
            best_train_loss = min(best_train_loss, avg_loss)

            # Compute validation metrics
            n_batches = 0
            valid_acc = 0
            valid_loss = 0

            with torch.no_grad():
                for (inputs, labels) in valid_data:

                    outputs = model(inputs)

                    accuracy_counter.update(outputs, labels.argmax(1))

                    valid_loss += loss_fn(outputs, labels).item()

                    n_batches += 1

            valid_acc = accuracy_counter.compute().item()
            valid_loss /= n_batches 

            iter.set_description(f"[Epoch: {i+1} / {n_epochs}] Train Acc: {round(avg_acc,4)} | Train Loss: {round(avg_loss,4)} | Val Acc: {round(valid_acc,4)} | Val Loss: {round(valid_loss,4)}")

            # Log to tensorboard
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : valid_loss },
                    i + 1)
            
            writer.add_scalars('Training vs. Validation Accuracy',
                    { 'Training' : avg_acc, 'Validation' : valid_acc },
                    i + 1)
            writer.flush()

            # Log if needed
            if self._use_wandb:
                wandb.log(
                    {
                    "training_loss" : avg_loss, 
                    "valid_loss" : valid_loss,
                    "training_acc" : avg_acc,
                    "valid_acc" : valid_acc 
                    }
                )

            # Keep track of best model
            best_acc = max(valid_acc, best_acc)
            best_loss = min(valid_loss, best_loss)
            # TODO save to disk the best model

        timer.stop()
        print(f"Training finished, best loss on validation: {best_loss}, best accuracy on validation {best_acc}")

        result = TrainResult(
            best_loss_on_validation=best_loss,
            best_acc_on_validation=best_acc,
            best_loss_on_train=best_train_loss,
            best_acc_on_train=best_train_acc,
            n_epochs=n_epochs,
            n_classes=self._valid_data._labels.shape[1],
            train_duration=timer.time_elapsed()
        )

        return result 

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

    def train_k_folds(self, 
                    model_class : Type[nn.Module], 
                    optimizer_fn : Callable[[], torch.optim.Optimizer],
                    model_args : Dict[str, Any], 
                    k_folds : int = 5, 
                    n_epochs : int = 10, 
                ) -> List[TrainResult]:
        """Run a k-folds training session.

        Args:
            k_folds (int, optional): _description_. Defaults to 5.
        """
        # Concat train and valid into one
        train_dataset = self._train_data
        valid_dataset = self._valid_data
        dataset = ConcatDataset([train_dataset, valid_dataset])

        # kfold: subsets of datasets
        torch.manual_seed(42)
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # Results for each fold
        results = []

        for fold_i, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
            
            print(f"-- < Fold {fold_i} > -----------------------")

            # Create samplers to select which rows will be used during training
            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)

            # Create dataloaders
            train_dataloader = DataLoader(dataset, batch_size=self._batch_size, sampler=train_subsampler)
            valid_dataloader = DataLoader(dataset, batch_size=self._batch_size, sampler=valid_subsampler)

            # Init model 
            # model = model_class(**model_args).to("cuda")
            self._model.apply(Trainer._reset_weights)

            # Init optimizer
            optim = optimizer_fn()

            # Run a training session with this configuration
            
            result = self.run_train(
                n_epochs, 
                valid_dataloader=valid_dataloader, 
                optim=optim, 
                train_dataloader=train_dataloader, 
                acceptance_th=self._acceptance_th,
                )
            results.append(result)

            # Print results so far TODO
            print(result.as_report())

        return results

    @staticmethod
    def _reset_weights(m : nn.Module):
        '''
            Try resetting model weights to avoid
            weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    @torch.no_grad()
    def compute_predictions(self) -> torch.Tensor:
        """Compute all predictions usign the provided validation data 

        Returns:
            torch.Tensor: vector with index of each predicted class per row.
            Size: (val_data.nrows)
        """
        results = torch.tensor([])
        dataloader = DataLoader(self._valid_data)
        model = self._model
        for (input, _) in dataloader:
            guess = model(input)
            results = torch.cat([results, guess])

        results = torch.argmax(results, axis=1)

        return results

    @torch.no_grad()
    def confusion_matrix(self) -> torch.Tensor:
        
        # Compute model predictions
        preds = self.compute_predictions()
        
        # Compute labels 
        d = DataLoader(self._valid_data, batch_size=1000000)
        labels = torch.tensor([])

        for (_, labs) in d:
            labels = labs
        
        _, n_classes = labels.shape
        labels = torch.argmax(labels, axis=1)

        assert labels.shape == preds.shape

        # Build pairs and confusion matrix itself\
        pairs = torch.stack([preds, labels], axis=1)
        conf_matrix = torch.zeros((n_classes, n_classes))

        for (y_pred, y) in pairs:
            conf_matrix[y_pred, y] += 1

        return conf_matrix

    def plot_confusion_matrix(self, cm : np.ndarray,  classes : np.ndarray, labelmap : Dict[int,str], normalize : bool = False, title : str ="Confusion Matrix", cmap = plt.cm.Blues):
        
        cm = np.array(cm.cpu())
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            cm.astype("int")
            print('Confusion matrix, without normalization')

        print(cm)
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, [labelmap[i] for i in classes], rotation=45)
        plt.yticks(tick_marks, [labelmap[i] for i in classes])

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

class ScopedTimer:
    """Simple timer that will count how much time is spend in the function where it's created
    """

    def __init__(self, timer_name : str) -> None:
        self._timer_name = timer_name
        self._start = datetime.now()
        self._running = True

    def stop(self) -> float:
        """Stop timer and return elapsed time

        Returns:
            float: Elapsed time in milliseconds (ms). -1 if already stopeed
        """

        if not self._running:
            return -1


        self._running = False
        self._end = datetime.now()

        time_ellapsed = self.time_elapsed()

        print(f"Time ellapsed for {self._timer_name}: {round(time_ellapsed,4)} s")

        return time_ellapsed

    def __del__(self):
        self.stop()

    def time_elapsed(self) -> float:
        """How many time has passed

        Returns:
            float: time passed in ms
        """ 

        start = self._start
        end = datetime.now() if self._running else self._end

        return (end - start).seconds