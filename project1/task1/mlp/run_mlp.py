import pandas as pd
from skimpy import clean_columns
from pickle import load
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
#from neural_additive_model import NAM
from icecream import ic
import os

# append path to parent folder to allow imports from utils folder
import sys

sys.path.append("\\utils\\utils.py")
from utils.utils import (
    set_all_seeds,
    HeartFailureDataset,
    train_and_validate,
    test,
    EarlyStopping,
)

#Note: Alt t should work, but doesn't see article
#Somehow the directory doesn't exist



SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 1_000


def main():
    ## Load the training and validation data
    train_df = pd.read_csv("../data/heart_failure/train_val_split.csv").pipe(
        clean_columns
    )
    X_train = train_df.drop(columns=["heart_disease"], axis=1)
    X_train = X_train.drop(366)
    y_train = train_df["heart_disease"]
    y_train = y_train[X_train.index]

    # create categorical variable for cholesterol level
    X_train["chol_level"] = pd.cut(
        X_train["cholesterol"],
        bins=[-1, 100, 200, 240, 1000],
        labels=["imputed", "normal", "borderline", "high"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train.to_numpy(), test_size=0.2, shuffle=True, random_state=SEED
    )

    ## Load the test data
    test_df = pd.read_csv("../data/heart_failure/test_split.csv").pipe(clean_columns)
    X_test = test_df.drop(columns=["heart_disease"], axis=1)
    # create categorical variable for cholesterol
    X_test["chol_level"] = pd.cut(
        X_test["cholesterol"],
        bins=[-1, 10, 200, 240, 1000],
        labels=["imputed", "normal", "borderline", "high"],
    )
    y_test = test_df["heart_disease"].to_numpy()

    # ----- Note Paul: I deactivated the preprocessor for now:
    # load and apply preprocessor:
    preprocessor = load(open("../models/preprocessor.pkl", "rb"))

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Create the dataset and dataloader
    train_dataset = HeartFailureDataset(X_train, y_train)
    val_dataset = HeartFailureDataset(X_val, y_val)
    test_dataset = HeartFailureDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True
    )

    model = NAM(
        n_features=X_train.shape[1],
        in_size=1,
        out_size=1,
        hidden_profile=[32, 64, 64, 32],
        use_exu=True,
        use_relu_n=True,
        within_feature_dropout=0.3,
        feature_dropout=0.0,
    ).to(DEVICE)
    # use BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=9.6e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.95)

    # visualize the progress in the tensorboard by typing
    # `tensorboard --logdir logs` in the terminal and then navigate
    # to the created process in the browser
    writer = SummaryWriter("logs/nam_experiment")

    ES = EarlyStopping("../models/neural_additive_model.pth")

    # Set seed for reproducibility
    set_all_seeds(SEED)

    _, _, _, _, best_threshold = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        n_epochs=N_EPOCHS,
        ES=ES,
        forward_returns_tuple=True,
        summary_writer=writer,
        scheduler=scheduler,
        device=DEVICE,
    )

    set_all_seeds(SEED)

    test(
        model,
        test_loader,
        criterion,
        forward_returns_tuple=True,
        device=DEVICE,
        threshold=best_threshold,
    )


if __name__ == "__main__":
    main()
