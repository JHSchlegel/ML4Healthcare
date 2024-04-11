# This file includes the utility functions for the neural additive model, CNN and
# the MLP:

# in the following, whenever we refer to the "NAM paper", we mean the following paper:
# https://arxiv.org/pdf/2004.13912.pdf

# -------------------------------
# Packages and Presets
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from tqdm import trange, tqdm
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_curve
import os
from typing import Tuple, List
from PIL import Image
import torchvision

from icecream import ic


# ==========================================================================
# General Utilities
# ==========================================================================


# -------------------------
# Loading in Pneumonia Images
# -------------------------
def get_pneumonia_images(
    folder_path: str,
    img_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the images and labels from the folder

    Args:
        folder_path (str): Path to the folder containing train/val/test images
        img_size (Tuple[int, int], optional): Rescaled size of images. Defaults to (256, 256).

    Returns:
        Tuple[List[Image], np.ndarray]: A tuple of the images as np.ndarray
                    and the labels as np.ndarray
    """
    images = []
    labels = []

    for i, subfolder in enumerate(["NORMAL", "PNEUMONIA"]):
        for file in tqdm(
            os.listdir(os.path.join(folder_path, subfolder)),
            desc=f"Reading {subfolder} {folder_path.split('/')[-1]} images",
        ):

            # read in images using PIL
            with Image.open(os.path.join(folder_path, subfolder, file)) as img:
                # resize image and make ensure that it has 3 channels/ is in RGB format
                img = img.resize(img_size).convert("RGB")
                images.append(img)

            labels.append(i)

    return images, np.array(labels)


# -------------------------
# Custom Datasets
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


class PneumoniaDataset(Dataset):
    def __init__(
        self,
        images: List[Image.Image],
        labels: np.ndarray,
        transforms: torchvision.transforms = None,
    ):
        """Create torch Dataset from images and labels

        Args:
            images (List[Image]): List of images
            labels (np.ndarray): List of labels
            transforms (torchvision.transforms): Transforms to apply to the images.
                Defaults to None.
        """
        self.images = images
        self.labels = torch.tensor(labels, dtype = torch.long)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


# -------------------------
# Early Stopping
# -------------------------
#!!! This Early Stopping class is inspired by the following stackoverflow post:
#!!! https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopping:
    def __init__(
        self,
        best_model_path: str = None,
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
            best_model_path (str): Where to save the best model. If best_model_path is
                None, the model weights are not saved. Defaults to None.
            start (int): After how many epochs should early stopping be checked.
                Defaults to 10.
            patience (int, optional): Number of epochs without improvement until
            training process is stopped early. Defaults to 20.
            epsilon (float, optional): Tolerance of improvement to avoid stopping
                solely because of numerical issues. Defaults to 1e-6.
            device (torch.device, optional): Device to run the model on.
                Defaults to torch.device( "cuda" if torch.cuda.is_available() else "cpu" ).
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
            if self.best_model_path is not None:
                torch.save(self.best_model, self.best_model_path)
        elif current_epoch > self.start:
            self.counter += 1  # stop training if no improvement in a long time
            if self.counter >= self.patience:
                return True
        return False


# -------------------------
# Train Loop
# -------------------------
def train_and_validate_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_penalized_BCE: bool = False,
    output_regularization: float = 0.0,
    l2_regularization: float = 0.0,
    scheduler: torch.optim.lr_scheduler = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[float, float, float, float]:
    """Train loop for the model. Returns the train loss, validation loss, validation f1 scores and
    validation balanced accuracie scores over time and the optimal threshold
    for class assignments (calculated on the validation set).

    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): DataLoader with training data
        val_loader (DataLoader): DataLo0ader with validation data
        optimizer (torch.optim.Optimizer): Optimizer to minimize loss with
        criterion (nn.Module): Loss function to use
        use_penalized_BCE (bool): Whether the penalized Binary Cross Entropy for
            NAMs should be used. Defaults to False.
        output_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        l2_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        summary_writer (SummaryWriter, optional): SummaryWriter to log results
            of the training. Can then be used to visualize the progress in the tensorboard
            by typing `tensorboard --logdir EXPERIMENT_DIRNAME` in the terminal and
            then navigating to the created process in the browser. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate swcheduler. Defaults to None.
        device (torch.device, optional): Device to train on. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        Tuple[float, float, float, float]:
            Returns the train loss, validation loss, validation f1 score and
            validation balanced accuracy.
    """
    # set model to train mode for dropout
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()

        if use_penalized_BCE:
            aggregated_logits, feature_logits = model(x.to(device))
            loss = penalized_binary_cross_entropy(
                model,
                aggregated_logits,
                feature_logits,
                y.to(device),
                output_regularization=output_regularization,
                l2_regularization=l2_regularization,
            )

        else:
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item()
    # avg training loss for epoch
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
            if use_penalized_BCE:
                aggregated_logits, feature_logits = model(x.to(device))
                # calculate validation loss
                loss = penalized_binary_cross_entropy(
                    model,
                    aggregated_logits,
                    feature_logits,
                    y.to(device),
                    output_regularization=output_regularization,
                    l2_regularization=l2_regularization,
                )
                y_pred.extend(
                    F.sigmoid(aggregated_logits.detach()).cpu().numpy().round()
                )
            else:
                logits = model(x.to(device))
                if logits.size(1) == 1:
                    y_pred.extend(F.sigmoid(logits.detach()).cpu().numpy().round().astype(int))
                else:
                    y_pred.extend(torch.argmax(F.softmax(logits.detach()).cpu(), dim = 1).numpy().round().astype(int))
                
                # calculate validation loss
                loss = criterion(logits, y.to(device))
            val_loss += loss.item()
        # avg valdation loss for epoch
        val_loss /= len(val_loader)

    # calculate validation f1 score and balanced accuracy
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return train_loss, val_loss, f1, bal_acc


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    ES: EarlyStopping,
    use_penalized_BCE: bool = False,
    output_regularization: float = 0.0,
    l2_regularization: float = 0.0,
    summary_writer: SummaryWriter = None,
    scheduler: torch.optim.lr_scheduler = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """Train loop for the model. Returns the train loss, validation loss, validation f1 scores and
        validation balanced accuracie scores over time.

    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): DataLoader with training data
        val_loader (DataLoader): DataLo0ader with validation data
        optimizer (torch.optim.Optimizer): Optimizer to minimize loss with
        criterion (nn.Module): Loss function to use
        n_epochs (int): Number of epochs to train for
        ES (EarlyStopping): EarlyStopping instance to use for early stopping
            and calculation of optimal threshold
        use_penalized_BCE (bool): Whether the penalized Binary Cross Entropy for
            NAMs should be used. Defaults to False.
        output_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        l2_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        summary_writer (SummaryWriter, optional): SummaryWriter to log results
            of the training. Can then be used to visualize the progress in the tensorboard
            by typing `tensorboard --logdir EXPERIMENT_DIRNAME` in the terminal and
            then navigating to the created process in the browser. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate swcheduler. Defaults to None.
        device (torch.device, optional): Device to train on. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        Tuple[List[float], List[float], List[float], List[float], float]:
            Returns the train loss, validation loss and validation f1 scores.
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
            # train and validate for one epoch
            train_loss, val_loss, f1, bal_acc = train_and_validate_one_epoch(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                use_penalized_BCE,
                output_regularization,
                l2_regularization,
                scheduler,
                device,
            )
            # save current metrics in a dictionary to add them to the progressbar
            dct["train_loss"] = train_loss
            dct["val_loss"] = val_loss
            dct["f1"] = f1
            dct["bal_acc"] = bal_acc

            # update progressbar
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

    return train_losses, val_losses, f1_scores, bal_accs


# -------------------------
# Test Loop
# -------------------------
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    use_penalized_BCE: bool = False,
    output_regularization: float = 0.0,
    l2_regularization: float = 0.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[float, float, float]:
    """Test loop for the model. Returns the test loss, f1 score and balanced accuracy.

    Args:
        model (nn.Module): A neural network model.
        test_loader (DataLoader): DataLoader object with test data.
        criterion (nn.Module):  Loss function.
        use_penalized_BCE (bool): Whether the penalized Binary Cross Entropy for
            NAMs should be used. Defaults to False.
        output_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        l2_regularization (float, optional): Regularization coefficient for
            penalized Binary Cross Entropy. Defaults to 0.0.
        device (torch.device): Device to run the model on.


    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray]: Tuple with test loss,
            f1 score and balanced accuracy, model probabilities and true y's
    """
    test_loss = 0.0
    y_true = []
    y_pred = []
    model_probs = []
    with torch.no_grad():
        # set model to eval mode to disable dropout during validation
        model.eval()
        for x, y in test_loader:
            y_true.extend(y.numpy())
            # calculate model probabilities
            if use_penalized_BCE:
                aggregated_logits, feature_logits = model(x.to(device))
                # calculate validation loss
                loss = penalized_binary_cross_entropy(
                    model,
                    aggregated_logits,
                    feature_logits,
                    y.to(device),
                    output_regularization=output_regularization,
                    l2_regularization=l2_regularization,
                )

                probs = F.sigmoid(aggregated_logits.detach()).cpu().numpy()
                model_probs.extend(probs)

                y_pred.extend((probs.round()).astype(float))

            else:
                logits = model(x.to(device))
                if logits.size(1) == 1:
                    # probabilities for the positive class
                    probs = F.sigmoid(logits.detach().cpu()).numpy()
                else:
                    # probabilities for the positive class
                    probs = F.softmax(logits.detach().cpu(), dim = 1)[:, 1].numpy()
                model_probs.extend(probs)

                y_pred.extend((probs.round()).astype(float))

                # calculate validation loss
                loss = criterion(logits, y.to(device))
            test_loss += loss.item()

        test_loss /= len(test_loader)

    # calculate test f1 score and test balanced accuracy
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Test Loss: {test_loss}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Balanced Accuracy: {bal_acc}")

    return test_loss, f1, bal_acc, np.array(model_probs), np.array(y_true)


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
#!!! https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
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

        # the original paper used a truncated normal with mean 4 and sdev=0.5
        # for ExU weight initialization and a truncated normal with mean 0 and
        # sdev=0.5 for bias initialization.
        # see: https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py
        # used +- 2 standard deviations for a and b as in the tensorflow implementation
        # of truncated normal
        nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5, a=3.0, b=5.0)
        nn.init.trunc_normal_(self.bias, mean=0.0, std=0.5, a=-1.0, b=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.bias) @ torch.exp(self.weights), 0, 1)


class LinearLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        """Custom Linear hidden layer to avoid dimension mismatch in the NAM
        model when using a linear layer.

        Args:
            in_size (int): Dimension of input tensor
            out_size (int): Dimension of output tensor
        """
        super(LinearLayer, self).__init__()
        self.in_size, self.out_size = in_size, out_size

        self.weights = Parameter(torch.Tensor(in_size, out_size))
        self.bias = Parameter(torch.Tensor(in_size))

        # the NAM paper code initialized the weights using a glorot / Xavier
        # uniform initialization and the biases again with a truncated Normal with sdev=0.5
        # See the Activation Layer class of the followign link for more info:
        # https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py
        nn.init.xavier_uniform_(self.weights)
        nn.init.trunc_normal_(self.bias, mean=0.0, std=0.5, a=-1.0, b=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # !!! We had a look at the ReLULayer class of the following repository to
        # !!! debug the dimension errors in the forward method:
        # !!! https://github.com/kherud/neural-additive-models-pt/blob/master/nam/model.py
        # !!! we then copied "return (x - self.bias) @ self.weights" from there
        return (x - self.bias) @ self.weights


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


# -------------------------------
# Calculate Number of Inputs
# -------------------------------
#!!! This function is based on the create_nam_model function of the code from the GitHub repository of the NAM paper:
#!!! see: https://github.com/google-research/google-research/blob/master/neural_additive_models/graph_builder.py
def get_n_units(
    X_train: np.ndarray,
    n_basis_functions: int = 1_000,
    units_multiplier: int = 2,
) -> List[int]:
    """Get the number of input units for each feature for the NAM model.

    Args:
        X_train (np.ndarray): Preprocessed training data
        n_basis_functions (int, optional): Number of basis functions. Defaults to 100.
        units_multiplier (int, optional): Multiplier by which number of unique
            values of feature get multiplied. Defaults to 2.

    Returns:
        List[int]: Input size for each feature for the NAM model
    """
    n_cols = X_train.shape[1]
    n_unique_values = [len(np.unique(X_train[:, i])) for i in range(n_cols)]

    return [min(n_basis_functions, i * units_multiplier) for i in n_unique_values]


# -------------------------------
# Custom Loss Function
# -------------------------------
#!!! This function is translated into PyTorch from the original TensorFlow implementation of the NAM paper:
#!!! from https://github.com/google-research/google-research/blob/master/neural_additive_models/graph_builder.py#L84
def feature_ouput_regularization(feature_logits: torch.Tensor) -> torch.Tensor:
    """Calculate penalty term for the output of the feature networks.

    Args:
        feature_logits (torch.Tensor): Concatenated feature logits.

    Returns:
        torch.Tensor: Penalty term for the output of the feature networks.
    """
    per_feature_norm = torch.norm(feature_logits, dim=0, p=2)
    return torch.sum(per_feature_norm) / len(per_feature_norm)


#!!! This function is translated into PyTorch from the original TensorFlow implementation of the NAM paper:
#!!! from https://github.com/google-research/google-research/blob/master/neural_additive_models/graph_builder.py#L84
def weight_decay_feature_params(model: nn.Module, num_networks: int) -> torch.Tensor:
    """Calculate penalty term for the weights of the feature networks.

    Args:
        model (nn.Module): NAM model.
        num_networks (int): Number of feature networks.

    Returns:
        torch.Tensor: Penalty term for the weights of the feature networks
    """
    # note that tf.nn.l2_loss divides the squared Euclidean/Frobenius norm by 2
    # for more information: https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    feature_weights_losses = [
        torch.norm(param, p=2) ** 2 / 2
        for name, param in model.named_parameters()
        # only penalize weights that require gradients; don't penalize biases
        if (param.requires_grad == True) and ("weights" in name)
    ]
    return torch.sum(torch.stack(feature_weights_losses)) / num_networks


#!!! This function is translated into PyTorch from the original TensorFlow implementation of the NAM paper:
#!!! from https://github.com/google-research/google-research/blob/master/neural_additive_models/graph_builder.py#L84
def penalized_binary_cross_entropy(
    model: nn.Module,
    aggregated_logits: torch.Tensor,
    feature_logits: torch.Tensor,
    y_true: torch.Tensor,
    output_regularization: float = 0.0,
    l2_regularization: float = 0.0,
) -> torch.Tensor:
    """Get penalized binary cross entropy loss for an NAM model instance.

    Args:
        model (nn.Module): NAM model
        aggregated_logits (torch.Tensor): Output of the NAM model/ aggregated logits.
        feature_logits (torch.Tensor): Concatenate feature logits.
        y_true (torch.Tensor): Batch labels
        output_regularization (float, optional): Regularization coefficient
            for feature logits. Defaults to 0.0.
        l2_regularization (float, optional): Regularization coefficient for
            feature network weights. Defaults to 0.0.

    Returns:
        torch.Tensor: Penalized binary cross entropy loss
    """
    loss = F.binary_cross_entropy_with_logits(aggregated_logits, y_true)
    regularization_loss = 0.0
    if output_regularization > 0.0:
        regularization_loss += (
            feature_ouput_regularization(feature_logits) * output_regularization
        )
    if l2_regularization > 0.0:
        num_networks = feature_logits.shape[1]
        regularization_loss += (
            weight_decay_feature_params(model, num_networks) * l2_regularization
        )
    return loss + regularization_loss
