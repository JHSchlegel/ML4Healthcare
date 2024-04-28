import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from cnn import CNN

# confusion matrix:
from sklearn.metrics import confusion_matrix

# append grandparent directory to path to make sure we can access the utils file
import sys

sys.path.append("../..")
from utils.utils import (
    get_pneumonia_images,
    PneumoniaDataset,
    set_all_seeds,
    train_and_validate,
    test,
    EarlyStopping,
)


SEED = 42
VALIDATION_SIZE = 0.2

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

N_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIMIZER = torch.optim.Adam

LEARNING_RATE = 1e-5 # low learning rate to avoid changing pretrained weights too much
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.75

EARLY_STOPPING_START = 100


def main():
    train_images, train_labels = get_pneumonia_images("../data/chest_xray/train")
    val_images, val_labels = get_pneumonia_images("../data/chest_xray/val")
    test_images, test_labels = get_pneumonia_images("../data/chest_xray/test")

    # Since there are so few images (16 in total) in the validation set,
    # we decided to first concatenate the train and validation images and then
    # in a second step split this concatenated dateset into train and enlargened
    # validation set. Further, we will make sure that the classes are similarly
    # unbalanced in the validation set as in the training set. In the current
    # validation set, this is not the case.
    train_images_concat = train_images + val_images
    train_labels_concat = np.concatenate([train_labels, val_labels])

    # overwrite names for memory efficiency
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images_concat,
        train_labels_concat,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=train_labels_concat,
    )

    # Note that the transformations below are the same as the ones from the following
    # kaggle post:
    # https://www.kaggle.com/code/teyang/pneumonia-detection-resnets-pytorch
    # See our EDA for a justification of these transformations
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor()]
    )

    # for more info about our custom PneunomiaDataset class see the utils file
    train_set = PneumoniaDataset(
        train_images, train_labels, transforms=train_transforms
    )
    val_set = PneumoniaDataset(val_images, val_labels, transforms=val_transforms)
    test_set = PneumoniaDataset(test_images, test_labels, transforms=test_transforms)

    # next, we craete the data loaders
    train_loader = DataLoader(
        train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # set all seeds for reproducibility
    set_all_seeds(SEED)

    # Tensorflow tutorial on how to deal with imbalanced data in DeepLearnning
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    num_0 = np.sum(train_labels == 0)
    num_1 = np.sum(train_labels == 1)
    weight = torch.tensor(
        [
            (1/num_0) * (num_0 + num_1) / 2.0,
            (1/num_1) * (num_0 + num_1) / 2.0
            
        ], dtype = torch.float32
    ).to(DEVICE)
    # create the model
    set_all_seeds(SEED)

    # create the model
    model = CNN().to(DEVICE)

    # we unfreeze all layers to make sure that also the convolutional layers are finetuned
    # on our small pneumonia dataset
    for x in model.resnet.parameters():
        x.requires_grad = True

    # instantiate loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight = weight)

    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    # visualize the progress in the tensorboard by typing
    # `tensorboard --logdir logs` in the terminal and then navigate
    # to the created process in the browser
    writer = SummaryWriter("logs/cnn_experiment")

    # instantiate early stopping class
    ES = EarlyStopping(
        best_model_path="../models/cnn_all_unfrozen.pth",
        start=EARLY_STOPPING_START,
        epsilon=0,
        patience=20,
        save_model_state_dict=True,
    )

    # train the model on the trainisng data and validate on the validation data
    # the best model is saved in the best_model_path using the ES object from above
    train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=N_EPOCHS,
        ES=ES,
        summary_writer=writer,
        device=DEVICE,
    )

    test_loss, test_f1_score, test_balanced_accuracy, model_probs, y_true = test(
        model, test_loader, criterion, device=DEVICE
    )

    print(confusion_matrix(y_true, model_probs.round()))


if __name__ == "__main__":
    main()
