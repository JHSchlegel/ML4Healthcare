import random
import numpy as np
import torch
# This script contains all the utility functions for the CNN model.

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import copy
import os


# =========================================================================== #
#                 Data Loading and Preprocessing Utilities                    #
# =========================================================================== #
class PTB_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================================== #
#                             General Utilities                               #
# =========================================================================== #
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


# !!! This Early Stopping class is inspired by the following stackoverflow post:
# !!! https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# !!! and by this kaggle notebook:
# !!! https://www.kaggle.com/code/megazotya/ecg-transformer/notebook
class EarlyStopping:
    def __init__(
            self,
            start: int = 50,
            patience: int = 10,
            epsilon: float = 1e-6,
            verbose: bool = False,
            mode: str = "min",
    ):
        self.start = start
        self.counter = 0
        self.patience = patience
        self.epsilon = epsilon

        self.verbose = verbose
        self.mode = mode

        # initialize objects of which best value will be tracked
        self.best_model = nn.Identity
        self.best_epoch = 0
        self.best_score = np.inf if mode == "min" else -np.inf

    def early_stop(self, model: nn.Module, metric: float, current_epoch: int) -> bool:
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
        if (metric < self.best_score + self.epsilon and self.mode == "min") or (
                metric > self.best_score - self.epsilon and self.mode == "max"
        ):
            # reset number of epochs without improvement
            self.counter = 0
            # update best model and best score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_score = metric
            self.best_epoch = current_epoch

        elif current_epoch > self.start:
            self.counter += 1  # stop training if no improvement in a long time
            if self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping at epoch {current_epoch}. Best score was {self.best_score:.4f} in epoch {self.best_epoch}."
                    )
                return True
        return False

    def save_best_model(self, model_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.best_model, model_path)

    def get_best_model(self) -> nn.Module:
        model = torch.load_state_dict(self.best_model)
        return model


# =========================================================================== #
#                    Model Training and Evaluation                            #
# =========================================================================== #
def train_one_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
) -> tuple[float, float, float, float]:
    """Train the model for one epoch

    Args:
        model (nn.Module): Model to train
        optimizer (optim.Optimizer): Optimizer to use
        criterion (nn.Module): Loss function
        train_loader (DataLoader): Train data loader
        device (torch.device): Device on which calculations are executed

    Returns:
        tuple[float, float, float, float]: Tuple of train loss, accuracy,
            balanced accuracy and f1 score
    """

    model.train()

    total_loss = 0.0
    y_preds = []
    y_true = []
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(seq)
        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        y_true.extend(label.cpu().numpy())
        y_preds.extend(F.softmax(output, dim=1).argmax(dim=1).cpu().numpy())

    # calculate metrics
    train_loss = total_loss / len(train_loader.dataset)
    train_acc = accuracy_score(y_true, y_preds)
    train_balanced_acc = balanced_accuracy_score(y_true, y_preds)
    train_f1_score = f1_score(y_true, y_preds, average=None)
    return train_loss, train_acc, train_balanced_acc, train_f1_score


def validate_one_epoch(
        model: nn.Module, criterion: nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float, float, float]:
    """Validate the model for one epoch

    Args:
        model (nn.Module): Model to validate
        criterion (nn.Module): Loss function to use
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device on which calculations are executed

    Returns:
        tuple[float, float, float, float]: tuple of validation loss, accuracy,
            balanced accuracy and f1 score
    """

    model.eval()

    total_loss = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for seq, label in val_loader:
            seq, label = seq.to(device), label.to(device)

            output = model(seq)

            loss = criterion(output, label)
            total_loss += loss.item()

            y_true.extend(label.cpu().numpy())
            y_preds.extend(F.softmax(output, dim=1).argmax(dim=1).cpu().numpy())

    # calculate metrics
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = accuracy_score(y_true, y_preds)
    val_balanced_acc = balanced_accuracy_score(y_true, y_preds)
    val_f1_score = f1_score(y_true, y_preds, average=None)
    return val_loss, val_acc, val_balanced_acc, val_f1_score


def train_and_validate(
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        best_model_path: str = "weights/transformer_pe.pth",
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        num_epochs: int = 100,
        ES: EarlyStopping = None,

):
    # make fancy progress bar
    with trange(num_epochs) as t:
        for epoch in t:
            dct = {}
            train_loss, train_acc, train_balanced_acc, train_f1_score = train_one_epoch(
                model, optimizer, criterion, train_loader, device
            )
            val_loss, val_acc, val_balanced_acc, val_f1_score = validate_one_epoch(
                model, criterion, val_loader, device
            )

            # update progress bar
            t.set_description(f"Training Transformer")
            t.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss,
                train_balanced_acc=train_balanced_acc,
                val_balanced_acc=val_balanced_acc,
            )

            # reduce learning rate
            if scheduler is not None:
              scheduler.step(val_loss)
            if ES.early_stop(model, val_balanced_acc, epoch):
              break
    ES.save_best_model(best_model_path)
    return modelS


def test(
        model: nn.Module,
        criterion: nn.Module,
        test_loader: DataLoader,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
) -> tuple[float, float, float, float]:
    """Test the model

    Args:
        model (nn.Module): Model to test
        criterion (nn.Module): Loss function to use
        test_loader (DataLoader): Test data loader
        device (torch.device, optional): Device on which calculations are executed. Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").

    Returns:
        tuple[float, float, float, float]: Tuple of predicted probabilities,
            predicted labels, true labels and test loss
    """
    model.eval()

    total_loss = 0
    model_probs = []
    y_preds = []
    y_true = []
    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)

            output = model(seq)

            loss = criterion(output, label)
            total_loss += loss.item()

            model_probs.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
            y_true.extend(label.cpu().numpy())
            y_preds.extend(F.softmax(output, dim=1).argmax(dim=1).cpu().numpy())


    # calculate metrics
    test_loss = total_loss / len(test_loader.dataset)
    test_acc = accuracy_score(y_true, y_preds)
    test_balanced_acc = balanced_accuracy_score(y_true, y_preds)
    #test_f1_score = f1_score(y_true, y_preds, average = None)
    #print(test_f1_score)
    print(
        f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}, Test balanced accuracy: {test_balanced_acc:.4f}S"
    )
    return model_probs, y_preds, y_true, test_loss