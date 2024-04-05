import pandas as pd
from skimpy import clean_columns
from pickle import load
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from neural_additive_model import NAM


# append path to parent folder to allow imports from utils folder
import sys

sys.path.append("../..")
from utils.utils import (
    set_all_seeds,
    HeartFailureDataset,
    train_and_validate,
    test,
    EarlyStopping,
    get_n_units,
    penalized_binary_cross_entropy,
)


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 1_000


def main():
    ## Load the training and validation data
    train_df = pd.read_csv("../data/heart_failure/train_val_split.csv").pipe(
        clean_columns
    )
    X_train = train_df.drop(columns=["heart_disease"], axis=1)
    outlier_idx = X_train.query("resting_bp == 0").index
    X_train = X_train.drop(outlier_idx)
    y_train = train_df["heart_disease"]
    y_train = y_train[X_train.index]

    # create categorical variable for cholesterol level
    X_train["chol_level"] = pd.cut(
        X_train["cholesterol"],
        bins=[-1, 10, 200, 240, 1000],
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
        in_size=get_n_units(X_train),
        out_size=1,
        hidden_profile=[64, 64, 32],
        use_exu=True,
        use_relu_n=True,
        within_feature_dropout=0.1,
        feature_dropout=0.1,
    ).to(DEVICE)
    # use BCEWithLogitsLoss for numerical stability
    criterion = penalized_binary_cross_entropy  # nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # visualize the progress in the tensorboard by typing
    # `tensorboard --logdir logs` in the terminal and then navigate
    # to the created process in the browser
    writer = SummaryWriter("logs/nam_experiment")

    ES = EarlyStopping(
        best_model_path="../models/neural_additive_model.pth",
        epsilon=0.0,
    )

    # Set seed for reproducibility
    set_all_seeds(SEED)

    _, _, _, _, best_threshold = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=N_EPOCHS,
        ES=ES,
        scheduler=scheduler,
        summary_writer=writer,
        device=DEVICE,
        use_penalized_BCE=True,
        output_regularization=1e-4,  # 0.0,
        l2_regularization=9.6e-5,
    )

    set_all_seeds(SEED)

    test(
        model,
        test_loader,
        criterion,
        device=DEVICE,
        threshold=best_threshold,
        use_penalized_BCE=True,
        output_regularization=0.0,
        l2_regularization=9.6e-5,
    )


if __name__ == "__main__":
    main()
