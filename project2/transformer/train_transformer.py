# Run this script to train the transformer model

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from imblearn.over_sampling import SMOTE

import yaml
from icecream import ic

from transformer_utils import (
    PTB_Dataset,
    EarlyStopping,
    set_all_seeds,
    init_parameters,
    train_and_validate,
    test,
)

from transformer import Transformer

with open("transformer_config.yaml", "r") as file:
    config = yaml.safe_load(file)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_all_seeds(config["general"]["seed"])


# =========================================================================== #
#                            Training and Validation                          #
# =========================================================================== #
def main():
    # Load data:
    train_df = pd.read_csv(config["paths"]["ptb_train"], header=None)
    test_df = pd.read_csv(config["paths"]["ptb_test"], header=None)

    # separate features and labels:
    X_train_full = train_df.iloc[:, :-1].to_numpy()
    y_train_full = train_df.iloc[:, -1].to_numpy()

    # 0-pad sequences to uniform, 190, length:
    X_test = test_df.iloc[:, :-1].to_numpy()
    y_test = test_df.iloc[:, -1].to_numpy()

    X_train_full = np.c_[X_train_full, np.zeros((X_train_full.shape[0], 3))]
    X_test = np.c_[X_test, np.zeros((X_test.shape[0], 3))]

    # split into train and validation set:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config["preprocessing"]["validation_size"],
        stratify=y_train_full,
        random_state=config["general"]["seed"],
    )

    #!!! copied from https://www.kaggle.com/code/megazotya/ecg-transformer/notebook
    # We observed that oversampling the minority class tended to improve
    # the model's performance by a tiny bit.
    if config["preprocessing"]["use_smote"]:
        sm = SMOTE(random_state=config["general"]["seed"])
        X_train, y_train = sm.fit_resample(X_train, y_train)
        ic(X_train.shape, y_train.shape)
        ic(np.unique(y_train, return_counts=True))

    train_loader = DataLoader(
        PTB_Dataset(X_train, y_train),
        batch_size=config["dataloader"]["train_batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PTB_Dataset(X_val, y_val),
        batch_size=config["dataloader"]["val_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        PTB_Dataset(X_test, y_test),
        batch_size=config["dataloader"]["test_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    model = Transformer(
        num_classes=config["transformer"]["num_classes"],
        input_size=config["transformer"]["input_size"],
        model_size=config["transformer"]["model_size"],
        num_heads=config["transformer"]["num_heads"],
        num_layers=config["transformer"]["num_encoder_layers"],
        d_ff=config["transformer"]["dim_feed_forward"],
        dropout=config["transformer"]["dropout"],
        transformer_activation=config["transformer"]["transformer_activation"],
        use_padding_mask=config["transformer"]["use_padding_mask"],
    )
    model = model.to(DEVICE)
    init_parameters(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["adamw"]["learning_rate"],
        weight_decay=config["adamw"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
    )

    early_stopping = EarlyStopping(
        start=config["early_stopping"]["start"],
        patience=config["early_stopping"]["patience"],
        verbose=True,
        mode="max",
    )
    summary_writer = SummaryWriter(log_dir=config["paths"]["summary_writer"])

    model = train_and_validate(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=config["paths"]["transformer_model"],
        device=DEVICE,
        num_epochs=config["transformer"]["num_epochs"],
        ES=early_stopping,
        summary_writer=summary_writer,
    )

    model_probs, model_preds, true_labels, test_loss = test(
        model=model, criterion=criterion, test_loader=test_loader, device=DEVICE
    )

    print(confusion_matrix(true_labels, model_preds))
    print(classification_report(true_labels, model_preds))


if __name__ == "__main__":
    main()
