# Machine Learning for Healthcare: Project 2

This repository contains the code for the first project of the course "Machine Learning for Healthcare" at ETH Zurich. 


- [Machine Learning for Healthcare: Project 2](#machine-learning-for-healthcare-project-2)
  - [Contributors](#contributors)
  - [Setup](#setup)
    - [Installation](#installation)
    - [Data](#data)
    - [Model Weights](#model-weights)
    - [Code Execution](#code-execution)
  - [Source Code Structure](#source-code-structure)
    - [Task 1:](#task-1)
    - [Task 2:](#task-2)
  - [Source Code Directory Tree](#source-code-directory-tree)
  - [Code References](#code-references)

## Contributors

- Paul Schlossmacher (23-941-636)
- Gianfranco Basile (23-955-545)
- Jan Schlegel (19-747-096)

## Setup
### Installation
To replicate our results we suggest setting up a conda environment (or any other virtual environment) and then installing the required packages:

```bash
conda create -n ml4hc python=3.10 pip
conda activate ml4hc
pip install -r requirements.txt
```

### Data
For part 1 and part 2 one has to access the data via a local path, since we decided to not upload it into our repository due to memory issues.

### Model Weights
For both part 1 and part 2, the model weights are stored in the `weights` directory of the respective folders.

### Code Execution
When executing the Jupyter Notebooks, make sure to set the current working directory correctly, so that the utils files can be imported.


## Source Code Structure
### Task 1:

- `EDA.ipynb`: Notebook for exploratory data analysis of the PTB dataset.
- `classical_ml`:
    - `classical_ml_config copy.yaml`: Contains hyperparameters for the classical_ml file
    - `classical_ml_config.yaml`: Contains hyperparameters for the classical_ml file
    - `lightgbm_hyperparameter_tuning.ipynb`: Hyperparameter tuning for lightgbm
    - `lightgbm_performance_eval.ipynb`: Performance evaluation for lightgbm
    - `logistic_regression_performance_eval.ipynb`: Performance evaluation for logistic regression
    - `random_forest_hyperparameter_tuning.ipynb`: Hyperparameter tuning for random forest
    - `random_forest_performance_eval.ipynb`: Performance evaluation for random forest
- `cnn`:
    - `cnn.ipynb`: Training and performance of the CNN with and without residual blocks
    - `cnn_utils.py`: Utils file for the CNN. Contains training functions etc.
- `rnn`:
    - `lstm.py`: Implementation, training and testing of the LSTM
    - `lstm_performance_evaluation.ipynb`: Evaluation of the performance of the LSTM
- `transformer`:
    - `train_transformer.py`: Contains hyperparameters for the classical_ml file
    - `transformer.ipynb`: Contains performance evaluation and general experiments for the transformer
    - `transformer.py`: Contains the Transformer class and the PositionalEncoding class
    - `transformer_config.yaml`: Contains hyperparameters for the transformer
    - `transformer_utils.py`: Utils file for the transformer. Contains training functions etc.
- `weights`: Contains the weights of our respective models

### Task 2:

- `transfer_learning`:
    - `cnn_transfer.ipynb`: Evaluation of the performance on the test set for the CNN 
    - `cnn_utils_transfer.ipynb`: utility functions for training and validating the CNN
- `representation_learning`:
    - `info_nce.py`: defintion of info nce loss function
    - `representation_learning.ipynb`: training and evaluation on the mitbih dataset of contrastive representation learning model
    - `representation_learning_config.yaml`: config file with all hyperparameters for the contrastive representation learning model
    - `representation_learning_utils.py`: utility functions for training and validating the contrastive representation learning model
- `visual_learn`:
    - `CNN_embedding_visualization.ipynb`: create the embeddings for Q1 and Q2 and plot the tsne and compare the distribution of different observations
    - `CNN_utils_visual.py`: utility functions for loading the data and create the embeddings
- `finetuning`:
    - `cnn_utils.py`: utility functions for training and performance evalutation of encoder based models
    - `fine_enc.ipynb`: Implementation and performance evaluation of encoder-based models
- `weights`: weights for the model implemented in part2

## Source Code Directory Tree
```bash
.
├── info                  # Project handout and presentation slides
├── task1                 # Folder for part 1 of the project
│   ├── classical_ml      # Folder for Q2
│   ├── cnn               # Folder for Q3
│   ├── rnn               # Folder for Q4
│   ├── transformer       # Folder for Q5
│   └── weights           # Folder of weights of all models
│   └── EDA               # EDA file
├── 
task2/
├── finetuning # Folder for Q4
├── representation_learning # Folder for Q2
├── transfer_learning # Folder for Q1
├── visual_learn # Folder for Q3
└── weights # Folder of weights of all models
```

## Code References
In our code files, we have included comments to cite references whenever we have incorporated, adapted, or been inspired by external code snippets.

