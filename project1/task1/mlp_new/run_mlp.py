import pandas as pd
from skimpy import clean_columns
from pickle import load
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from mlp import MLP
#from icecream import ic
import os
from sklearn.metrics import confusion_matrix

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
N_EPOCHS = 500

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 300

HIDDEN_PROFILE = [1024]
#USE_EXU = False
#USE_RELU_N = False
#WITHIN_FEATURE_DROPOUT = 0.4
#FEATURE_DROPOUT = 0.0


LEARNING_RATE = 0.003

SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.9

OPTIMIZER = torch.optim.Adam
CRITERION = penalized_binary_cross_entropy

#OUTPUT_REGULARIZATION = 0.0058
#L2_REGULARIZATION = 3.87e-5

EARLY_STOPPING_START = 60



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
        X_train, y_train.to_numpy(), test_size=0.2, shuffle=True,
        random_state=SEED, stratify=y_train
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

    
    model = MLP(
        in_size=25,
        out_size=1,
        hidden_profile=[32, 64, 64, 32],
        dropout=0.3
    ).to(DEVICE)
    
    # use BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=9.6e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.95)

    # visualize the progress in the tensorboard by typing
    # `tensorboard --logdir logs` in the terminal and then navigate
    # to the created process in the browser
    
    writer = SummaryWriter("logs/mlp_experiment")

    ES = EarlyStopping(best_model_path="../models/mlp.pth")

    # Set seed for reproducibility
    set_all_seeds(SEED)

    #_, _, _, _, best_threshold = train_and_validate(
    train_and_validate(
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
        use_penalized_BCE=False
        #forward_returns_tuple=True
        #forward_returns_tuple=False
    )

    set_all_seeds(SEED)

    #test(
    test_loss, test_f1_score, balanced_accuracy, model_probs, y_true = test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        #forward_returns_tuple=True,
        #forward_returns_tuple=False,
        #threshold=best_threshold,
    )
    
    # confusion matrix:
    print(confusion_matrix(y_true, model_probs.round()))

if __name__ == "__main__":
    main()
