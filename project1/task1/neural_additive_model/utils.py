# This file includes the utility functions for the neural additive model:

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
from torchsummary import summary
from sklearn.metrics import f1_score, balanced_accuracy_score
import os


# TODO: fix trunc_norm warning: mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.
# TODO: include early stopping
# TODO summarizer
# ==========================================================================
# General Utilities
# ==========================================================================


# -------------------------
# Custom Dataset
# -------------------------
class HeartFailureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler,
    n_epochs: int,
    name: str,
    device: torch.device,
) -> float:
    writer = SummaryWriter(f"logs/{name}")
    with trange(n_epochs) as t:
        for i in t:
            dct = {}
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x.to(device))
                loss = criterion(y_pred, y.to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            dct["train_loss"] = train_loss

            val_loss = 0.0
            y_true = []
            y_pred = []
            with torch.no_grad():
                model.eval()
                for x, y in val_loader:
                    y_true.extend(y.numpy())
                    y_pred.extend(F.sigmoid(model(x.to(device)).detach()).cpu().numpy())
                    loss = criterion(model(x.to(device)), y.to(device))
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            f1 = f1_score(y_true, np.round(y_pred))
            bal_acc = balanced_accuracy_score(y_true, np.round(y_pred))

            t.set_postfix(
                train_loss=train_loss, val_loss=val_loss, f1=f1, bal_acc=bal_acc
            )

            writer.add_scalar("train_loss", train_loss, i)
            writer.add_scalar("val_loss", val_loss, i)
            writer.add_scalar("f1", f1, i)
            writer.add_scalar("bal_acc", bal_acc, i)


from icecream import ic


def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    test_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            ic(f"{type(x)} {type(y)}")
            y_true.extend(y.numpy())
            y_pred.extend(F.sigmoid(model(x.to(device)).detach()).cpu().numpy())
            loss = criterion(model(x.to(device)), y.to(device))
            test_loss += loss.item()
        test_loss /= len(test_loader)

    f1 = f1_score(y_true, np.round(y_pred))
    bal_acc = balanced_accuracy_score(y_true, np.round(y_pred))

    print(f"Test Loss: {test_loss}")
    print(f"F1 Score: {f1}")
    print(f"Balanced Accuracy: {bal_acc}")

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
        super(ExULayer, self).__init__()
        self.in_size, self.out_size = in_size, out_size

        self.weights = Parameter(torch.Tensor(in_size, out_size))
        self.bias = Parameter(torch.Tensor(in_size))

        # the original paper used a truncated normal with mean 4 and sd 0.5
        # for ExU weight initialization.
        # see: https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py
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
        super(ReLUn, self).__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(F.relu(x), min=0, max=self.n)
