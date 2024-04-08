import pandas as pd
from skimpy import clean_columns
from pickle import load
import os
cwd = os.getcwd()
cwd
#%%
# Load the training data
train_dir = os.path.join(cwd, 'task1','data','heart_failure','train_val_split.csv')
## Load the training and validation data
train_df = pd.read_csv(train_dir).pipe(
        clean_columns)
X_train = train_df.drop(columns=["heart_disease"], axis=1)
X_train = X_train.drop(366)
y_train = train_df["heart_disease"]
y_train = y_train[X_train.index]

# create categorical variable for cholesterol level
X_train["chol_level"] = pd.cut(
    X_train["cholesterol"],
    bins=[-1, 100, 200, 240, 1000],
    labels=["imputed", "normal", "borderline", "high"]
)
#%%
## Load the test data
test_dir = os.path.join(cwd, 'data','heart_failure','test_split.csv')
test_df = pd.read_csv(test_dir).pipe(clean_columns)
X_test = test_df.drop(columns=["heart_disease"], axis=1)
# create categorical variable for cholesterol
X_test["chol_level"] = pd.cut(
    X_test["cholesterol"],
    bins=[-1, 10, 200, 240, 1000],
    labels=["imputed", "normal", "borderline", "high"]
)
y_test = test_df["heart_disease"].to_numpy()
#%%
# load and apply preprocessor:
prep_dir = os.path.join(cwd, 'models','preprocessor.pkl')
preprocessor = load(open(prep_dir, "rb"))
#%%
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%%
X_train.head()
#%%
#%%

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%



#%%
#%%

