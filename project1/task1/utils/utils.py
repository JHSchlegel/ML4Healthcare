# This file includes the utility functions for the neural additive model and
# the MLP:

# -------------------------------
# Packages and Presets
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from tqdm import trange
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_curve
import os
from typing import Tuple, List


# ==========================================================================
# General Utilities
# ==========================================================================


# -------------------------
# Custom Dataset
# -------------------------
class HeartFailureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Create torch Dataset from preprocessed features and labels

        Args:
            X (np.ndarray): train/validation/test features
            y (np.ndarray): train/validation/test labels
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Find Best Threshold
# -------------------------
#!!! inspired by the following data science stack exchange post:
# https://datascience.stackexchange.com/questions/96690/how-to-choose-the-right-threshold-for-binary-classification
def get_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Get the optimal threshold for class assignments. Can then be used to assign
    an observation to class 0 if probs <= optimal threshold and to class 1 if
    probs > optimal threshold.

    Args:
        probs (np.ndarray): Predicted probabilities for each class
        y_true (np.ndarray): True labels

    Returns:
        float: Optimal threshold
    """
    precision, recall, thresh = precision_recall_curve(y_true, probs)

    # calculate f1 score for every threshold
    f1 = (2 * precision * recall) / (precision + recall)
    return thresh[np.argmax(f1)]


# -------------------------
# Early Stopping
# -------------------------
#!!! This Early Stopping class is inspired by the following stackoverflow post:
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopping:
    def __init__(
        self,
        best_model_path: str,
        start: int = 50,
        patience: int = 20,
        epsilon: float = 1e-6,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Class that implements early stopping if there was no improvement during
        training for a sufficiently long time and calculates the optimal threshold
        for class assignments.

        Args:
            best_model_path (str): Where to save the best model
            start (int): After how many epochs should early stopping be checked.
                Defaults to 10.
            patience (int, optional): Number of epochs without improvement until
            training process is stopped early. Defaults to 20.
            epsilon (float, optional): Tolerance of improvement to avoid stopping
                solely because of numerical issues. Defaults to 1e-6.
            device (torch.device, optional): Device to run the model on. Defaults to torch.device( "cuda" if torch.cuda.is_available() else "cpu" ).
        """
        self.best_model_path = best_model_path
        self.start = start
        self.counter = 0
        self.patience = patience
        self.epsilon = epsilon
        self.best_model = nn.Identity
        self.best_val_loss = np.inf
        self.device = device

    def early_stop(self, val_loss: float, model: nn.Module, current_epoch: int) -> bool:
        """Whether training should be stopped or not. If there was no improvement
        in a long time, training should be stopped. Continuously saves the best
        model.

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model
            current_epoch (int): Current epoch number

        Returns:
            bool: Whether training should be stopped or not
        """
        # check whether improvement was large enough (if there was any at all)
        if val_loss < self.best_val_loss + self.epsilon:
            # reset number of epochs without improvement
            self.counter = 0
            # update best model and best validation loss
            self.best_model = model
            self.best_val_loss = val_loss
            # save current best model
            torch.save(self.best_model.state_dict(), self.best_model_path)
        elif current_epoch > self.start:
            self.counter += 1  # stop training if no improvement in a long time
            if self.counter >= self.patience:
                return True
        return False

    def _get_best_threshold(self, val_loader: DataLoader) -> float:
        """Get the optimal threshold for class assignments. Can then be used to assign
        an observation to class 0 if probs <= optimal threshold and to class 1 if
        probs > optimal threshold.

        Args:
            val_loader (DataLoader): Validation DataLoader containing validation
            data

        Returns:
            float: Optimal threshold
        """
        y_true = []
        model_probabilities = []
        with torch.no_grad():
            # set model to eval mode to disable dropout during validation
            self.best_model.eval()
            for x, y in val_loader:
                # add true labels and predicted model probabilities to list
                y_true.extend(y.numpy())
                model_probabilities.extend(
                    F.sigmoid(self.best_model(x.to(self.device)).detach()).cpu().numpy()
                )
        return get_best_threshold(np.array(model_probabilities), np.array(y_true))


# -------------------------
# Train Loop
# -------------------------
def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    ES: EarlyStopping,
    summary_writer: SummaryWriter = None,
    scheduler: torch.optim.lr_scheduler = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """Train loop for the model. Returns the train loss, validation loss, validation f1 scores and
    validation balanced accuracie scores over time and the optimal threshold
    for class assignments (calculated on the validation set).

    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): DataLoader with training data
        val_loader (DataLoader): DataLo0ader with validation data
        optimizer (torch.optim.Optimizer): Optimizer to minimize loss with
        criterion (nn.Module): Loss function to use
        n_epochs (int): Number of epochs to train for
        ES (EarlyStopping): EarlyStopping instance to use for early stopping
            and calculation of optimal threshold
        summary_writer (SummaryWriter, optional): SummaryWriter to log results
            of the training. Can then be used to visualize the progress in the tensorboard
            by typing `tensorboard --logdir EXPERIMENT_DIRNAME` in the terminal and
            then navigating to the created process in the browser. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate swcheduler. Defaults to None.
        device (torch.device, optional): Device to train on. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        Tuple[List[float], List[float], List[float], List[float], float]:
            Returns the train loss, validation loss, validation f1 scores and
            validation balanced accuracie scores over time and the optimal threshold
            for class assignments (calculated on the validation set).
    """

    # initialize metrics
    train_losses = []
    val_losses = []
    f1_scores = []
    bal_accs = []

    # fancy progress bar
    with trange(n_epochs) as t:
        for i in t:
            dct = {}
            # set model to train mode for dropout
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()

                y_pred = model(x.to(device))

                loss = criterion(y_pred, y.to(device))
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
            # avg training loss
            train_loss /= len(train_loader)

            val_loss = 0.0
            y_true = []
            y_pred = []
            with torch.no_grad():
                # set model to eval mode to disable dropout during validation
                model.eval()
                for x, y in val_loader:
                    # add true and predicted labels for balanced accuracy and f1 score
                    y_true.extend(y.numpy())
                    # for predicted labels: use default threshold of 0.5
                    # threshold is tuned for the best model later on
                    y_pred.extend(
                        F.sigmoid(model(x.to(device)).detach()).cpu().numpy().round()
                    )

                    # calculate validation loss
                    loss = criterion(model(x.to(device)), y.to(device))
                    val_loss += loss.item()
                # avg valdation loss
                val_loss /= len(val_loader)

            # calculate validation f1 score and balanced accuracy
            f1 = f1_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)

            # save current metrics in a dictionary to add them to the progressbar
            dct["train_loss"] = train_loss
            dct["val_loss"] = val_loss
            dct["f1"] = f1
            dct["bal_acc"] = bal_acc

            t.set_postfix(dct)

            # log metrics to tensorboard:
            if summary_writer is not None:
                summary_writer.add_scalar("train_loss", train_loss, i)
                summary_writer.add_scalar("val_loss", val_loss, i)
                summary_writer.add_scalar("f1", f1, i)
                summary_writer.add_scalar("bal_acc", bal_acc, i)

            # append metrics:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            f1_scores.append(f1)
            bal_accs.append(bal_acc)

            # check whether training should be stopped early:
            if ES.early_stop(val_loss, model, i):
                break
    # get optimal threshold for class assignment for the best model;
    best_threshold = ES._get_best_threshold(val_loader)
    return train_losses, val_losses, f1_scores, bal_accs, best_threshold


# -------------------------
# Test Loop
# -------------------------
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Test loop for the model. Returns the test loss, f1 score and balanced accuracy.

    Args:
        model (nn.Module): A neural network model.
        test_loader (DataLoader): DataLoader object with test data.
        criterion (nn.Module):  Loss function.
        device (torch.device): Device to run the model on.
        threshold (float, optional): Optimal threshold for class assignment calculated
        on the validation set. Defaults to 0.5.

    Returns:
        Tuple[float, float, float]: Tuple with test loss, f1 score and balanced accuracy.
    """
    test_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        # set model to eval mode to disable dropout during validation
        model.eval()
        for x, y in test_loader:
            y_true.extend(y.numpy())
            # calculate model probabilities
            probs = F.sigmoid(model(x.to(device)).detach()).cpu().numpy()
            # use optimal threshold to assign class
            y_pred.extend((probs >= threshold).astype(float))

            loss = criterion(model(x.to(device)), y.to(device))
            test_loss += loss.item()

        test_loss /= len(test_loader)

    # calculate test f1 score and test balanced accuracy
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Test Loss: {test_loss}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Balanced Accuracy: {bal_acc}")

    return test_loss, f1, bal_acc


# -------------------------------
# Random Seed Initialization
# -------------------------------
# see https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
# for more information aboutreproducibility in pytorch
def set_all_seeds(seed: int):
    """Set all possible seeds to ensure reproducibility and to avoid randomness
    involved in GPU computations.

    Args:
        seed (int): Seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================================
# Neural Additive Model Specific Utilities
# ==========================================================================


# -------------------------------
# Custom Hidden Layer
# -------------------------------
#!!! We used and adapted the following tutorial on writing custom layers in pytorch:
# https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
class ExULayer(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        """Exu hidden layer as described in the paper "Neural Additive Models:
        Interpretable Machine Learning with Neural Nets"
        (see: https://arxiv.org/pdf/2004.13912.pdf)

        Args:
            in_size (int): Dimension of input tensor
            out_size (int): Dimension of output tensor
        """
        super(ExULayer, self).__init__()
        self.in_size, self.out_size = in_size, out_size

        self.weights = Parameter(torch.Tensor(in_size, out_size))
        self.bias = Parameter(torch.Tensor(in_size))

        # the original paper used a truncated normal with mean 4 and sd 0.5
        # for ExU weight initialization.
        # see: https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py
        # used +- 2 standard deviations for a and b as in the tensorflow implementation
        # of truncated normal
        nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5, a=3.0, b=5.0)
        # if nothing specified (as in the github code of the nam paper (cf link above))
        # the bias is initialized with zeros in tensorflow, see:
        # https://stackoverflow.com/questions/40708169/how-to-initialize-biases-in-a-keras-model#:~:text=Weight%20and%20bias%20initialization%20for,bias_initializer%3D'zeros'%20are%20applied.
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.bias) @ torch.exp(self.weights)


# -------------------------------
# Custom Activation Function
# -------------------------------
class ReLUn(nn.Module):
    def __init__(self, n: float = 1.0):
        """ReLUn activation function as described in the paper "Neural Additive Models:
        Interpretable Machine Learning with Neural Nets"
        (see: https://arxiv.org/pdf/2004.13912.pdf). THis activation function
        makes it easier to model sharp jumps.

        Args:
            n (float, optional): Where to cut off ReLU. Defaults to 1.0.
        """
        super(ReLUn, self).__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), min=0, max=self.n)
