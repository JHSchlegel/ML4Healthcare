# Machine Learning for Healthcare: Project 1 

This repository contains the code for the first project of the course "Machine Learning for Healthcare" at ETH Zurich. 


- [Machine Learning for Healthcare: Project 1](#machine-learning-for-healthcare-project-1)
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
For part 1, the data is stored in the `data` directory of the `task1` folder. For part 2, you can download the pneumonia dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and create a directory `data` in the `task2` directory. Place the downloaded Pneumonia data in the `data` directory.

### Model Weights
For part 1, the model weights are stored in the `models` directory of the `task1` folder. For part 2, you can download the model weights from [Polybox](https://polybox.ethz.ch/index.php/s/HnIVNu2977hSQzV) using password `ml4hc`. After downloading these weights, create a directory `models` in the `task2` directory and place the weights in there.

### Code Execution
Before executing the `run_*.py` files, make sure to navigate to the respective directory. For example, to execute the MLP code, navigate to the `mlp` directory in the `task1` folder and only then run the `run_mlp.py` file. The reason for this is that the code uses relative paths to load the data and model weights and that we make use of utility functions defined in a utils folder of the parent directory.


## Source Code Structure
### Task 1:

- `eda.ipynb`: Notebook for exploratory data analysis of the heart failure dataset.
- `logistic_lasso.ipynb`: Notebook for training logistic lasso regression and analyzing its feature importances.
- `neural_additive_model/`: Contains all files related to the neural additive model (NAM).
  - `run_nam.py`: Script to train the NAM.
  - `nam.py`: Script defining the NAM model class.
  - `hyperparameter_tuning.ipynb`: Notebook to perform hyperparameter tuning for the NAM.
  - `nam_experiments.ipynb`: Notebook for analyzing the performance of the NAM and visualizing the feature importances
- `mlp/`: Contains all files related to the Multi-Layer Perceptron (MLP) model.
  - `run_mlp.py`: Script to train the MLP model.
  - `mlp.py`: Script defining the MLP model class.
  - `mlp_performance.ipynb`: Notebook for evaluating MLP model performance.
  - `mlp_shap.ipynb`: Notebook for SHAP value analysis of the MLP model.
  
### Task 2:

- `eda.ipynb`: Notebook for exploratory data analysis of the pneumonia dataset.
- `grad_cam.ipynb`: Notebook containing Grad-CAM analysis for CNN trained on original labels
- `int_grad.ipynb`: Notebook containing Integrated Gradients analysis for CNN trained on original labels
- `cnn/`: Contains all files related to the Convolutional Neural Network (CNN) model.
  - `run_cnn.py`: Script to train the CNN model.
  - `cnn.py`: Script defining the CNN model class.
  - `cnn_experiments.ipynb`: Notebook for evaluating CNN model performance and visualizing example images.
- `data_randomization_test/`: Contains all files related to the data randomization tests.
  - `run_cnn_randomized.py`: Script to train the CNN model on permuted labels.
  - `cnn_performance_comparison.ipynb`: Notebook for comparing the performance of the CNN trained on original vs permuted labels.
  - `grad_cam_permuted_label.ipynb`: Notebook containing Grad-CAM analysis for CNN trained on permuted labels
  - `int_grad_permuted_label.ipynb`: Notebook containing Integrated Gradients analysis for CNN trained on permuted labels

## Source Code Directory Tree
```bash
.
├── info                                # Project handout and presentation slides
├── task1                               # Folder for part 1 of the project
│   ├── data                                # Data folder for part 1
│   │   └── heart_failure
│   ├── mlp                                 # Folder for creating and training the MLP as well as analyzing the performance/ shap values
│   ├── models                              # Model weights and preprocessor
│   └── neural_additive_model               # Folder for creating and training the NAM as well as analyzing the performance/ shap values
├── task2                               # Folder for part 2 of the project
│   ├── cnn                                 # Folder for creating and training the CNN (on the original labels) as well as analyzing the performance
│   └── data_randomization_test             # Folder for the analysis of GradCAM and Integrated Gradients for a CNN trained on permuted labels. Moreover contains the code to train the CNN on permuted labels and to compare the performance of the CNN trained on the original vs permuted labels
└── utils                               # Folder for global utility functions that are used by part 1 and part 2
```
## Code References
In our code files, we have included comments to cite references whenever we have incorporated, adapted, or been inspired by external code snippets.
