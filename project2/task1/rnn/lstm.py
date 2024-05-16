# %%
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List
from tqdm import trange, tqdm
from multiprocessing import cpu_count
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score
from icecream import ic

from imblearn.over_sampling import SMOTE

import os
import logging

logging.basicConfig(level=logging.INFO, filename="rnn.log")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
input_dim = 1  # x_train.shape[1]

batch_size = 128
hidden_dim = 128
layer_dim = 2
output_dim = 1
num_epochs = 200
LR = 3e-4


class LSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, layer_dim, target_size, bidirectional=False
    ):
        super().__init__()
        self.input_dim = input_dim  # Size of the imput
        self.hidden_dim = hidden_dim  # Number ho hidden size
        self.layer_dim = layer_dim  # Number of hidden layer
        self.target_size = target_size  # Number of size of the output
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=False,
            dropout=0.2,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * (1 + bidirectional) * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, input):
        # ic(input.shape)
        input = input.unsqueeze(2).permute(1, 0, 2)

        # ic(input.shape)
        out, _ = self.lstm(input)
        # pooled = self.pool(out.transpose(1, 2)).squeeze(-1)
        out = torch.concat([out.mean(dim=0), out.max(dim=0).values], dim=1)
        logits = self.fc(out)
        scores = F.sigmoid(logits)
        return scores


# %%
# os.chdir(r"C:\Users\gianf\Desktop\ETH\II_Semestre\ML4HC\ML4Healthcare\project2\RNN")
path_train = r"../../data/ptbdb_train.csv"
path_test = r"../../data/ptbdb_test.csv"

ptbdb_train = pd.read_csv(path_train, header=None)
ptbdb_test = pd.read_csv(path_test, header=None)

x_train = ptbdb_train.iloc[:, :-1].values
y_train = ptbdb_train.iloc[:, -1].values

sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)


# %%


def train_and_validate(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    scheduler: _LRScheduler = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    best_bal_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            logits = model(inputs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # log summary statistics of gradinets, i.e min, max, number of zeros
            max_var = 0.0
            min_var = 0.0
            zeros = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    max_var = max(param.grad.max().item(), max_var)
                    min_var = min(param.grad.min().item(), min_var)
                    zeros += torch.sum(param.grad == 0).item()
            logging.info(f"max grad: {max_var}, min grad: {min_var}, zeros: {zeros}")
        train_loss /= len(train_loader)

        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for inputs, labels in valloader:
                labels = labels.unsqueeze(1)
                y_true.extend(labels.numpy())
                logits = model(inputs.to(device))
                y_pred.extend(logits.detach().cpu().numpy().round().astype(int))
                # print(y_pred)
                # calculate validation loss
                loss = criterion(logits, labels.to(device))
                val_loss += loss.item()
            # avg valdation loss for epoch
            val_loss /= len(val_loader)
        # calculate validation f1 score and balanced accuracy
        f1 = f1_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        if epoch > 19:
            if scheduler is not None:
                scheduler.step(val_loss)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), r"lstm_unidirectional.pth")
        # print(y_pred[0:60])
        print(
            f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Balanced Accuracy: {bal_acc:.4f},"
        )


# %%

model = LSTM(input_dim, hidden_dim, layer_dim, output_dim, bidirectional=False).to(
    DEVICE
)
# model.load_state_dict(torch.load("best_model_new.pth"))
# %%

# Split train in train and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)


x_test = ptbdb_test.iloc[:, :-1].values
y_test = ptbdb_test.iloc[:, -1].values

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoader

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
loss_function = nn.BCELoss()
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_and_validate(
    model=model,
    trainloader=train_loader,
    valloader=val_loader,
    n_epochs=num_epochs,
    optimizer=optim,
    criterion=loss_function,
    scheduler=None,  # torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=5,factor=0.5),
    device=DEVICE,
)
# %%


# %%
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[float, float, float]:
    test_loss = 0.0
    y_true = []
    y_pred = []
    model_probs = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            labels = labels.unsqueeze(1)
            y_true.extend(labels.numpy())
            logits = model(inputs.to(device))
            # logits = logits.detach().cpu().numpy().round().astype(int)
            # probs = F.sigmoid(logits)
            y_pred.extend(logits.detach().cpu().numpy().round().astype(int))
            # model_probs.extend(probs)
            # print(probs)
            # print(y_pred)
            # calculate validation loss
            loss = criterion(logits, labels.to(device))
            test_loss += loss.item()
            # avg valdation loss for epoch
        test_loss /= len(test_loader)
        # calculate validation f1 score and balanced accuracy
    f1 = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Test Loss: {test_loss}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Balanced Accuracy: {bal_acc}")

    # return test_loss, f1, bal_acc, np.array(model_probs), np.array(y_true)


# %%
model.load_state_dict(torch.load("lstm_unidirectional.pth"))
# %%
model.eval()
# %%
loss_function = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test(model, test_loader, loss_function, DEVICE)
# %%
# %%
