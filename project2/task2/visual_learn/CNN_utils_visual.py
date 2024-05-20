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


def get_embeddings(model, data_loader, device):
    """
    Obtain embeddings for the given dataset using the provided model.

    Parameters:
    - model: The PyTorch model used to generate embeddings.
    - data_loader: DataLoader providing the dataset.
    - device: The device on which the model and data are loaded.

    Returns:
    - embeddings: A numpy array containing the concatenated embeddings.
    """
    # Initialize an empty list to store the embeddings
    embeddings = []

    # Ensure the model is in evaluation mode
    model.eval()

    # No need to track gradients for this
    with torch.no_grad():
        # Iterate over the dataset
        for inputs, _ in data_loader:
            # Move the inputs to the same device as the model
            inputs = inputs.to(device)

            # Get the embeddings for this batch
            embedding = model(inputs)

            # Append the embeddings to the list
            embeddings.append(embedding.cpu().numpy())

    # Convert the list of embeddings to a single numpy array
    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings
