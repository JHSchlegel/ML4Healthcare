import pandas as pd
import numpy as np
from skimpy import clean_columns
from pickle import load
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
cwd = os.getcwd()
cwd

# %%
## Load the training and validation data
base_path = "C:/Users/gianf/Desktop/ETH/II_Semestre/ML4HC/ML4Healthcare/project1/task1"
train_df = pd.read_csv(base_path + "/data/heart_failure/train_val_split.csv").pipe(
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

test_df = pd.read_csv(base_path + "/data/heart_failure/test_split.csv").pipe(clean_columns)
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
preprocessor = load(open(base_path + "/models/preprocessor.pkl", "rb"))
#%%
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#%%
Lasso = LogisticRegressionCV(cv=10, random_state=0).fit(X_train, y_train)
#%%

#%%# get names of the transformed features
numeric_feature_names = preprocessor.named_transformers_[
    "numeric"
].get_feature_names_out()
categorical_feature_names = preprocessor.named_transformers_[
    "categorical"
].get_feature_names_out()
names = np.concatenate([numeric_feature_names, categorical_feature_names])
print(f"{names=}")
#%%
names_cleaned = [
    "Age",
    "Resting Blood Pressure",
    "Cholesterol",
    "Maximum Heart Rate",
    "Old Peak",
    "Sex Female",
    "Sex Male",
    "Chest Pain: Type Asymptomatic",
    "Chest Pain: Type Atypical Angina",
    "Chest Pain: Type Non-Anginal Pain",
    "Chest Pain: Type Typical Angina",
    "Fasting Blood Sugar < 120 mg/dl",
    "Fasting Blood Sugar > 120 mg/dl",
    "Resting ECG: Left Ventricular Hypertrophy",
    "Resting ECG: Normal",
    "Resting ECG: ST-T Wave Abnormality",
    "Exercise-Induced Angina: No",
    "Exercise-Induced Angina: Yes",
    "ST Slope: Downsloping",
    "ST Slope: Flat",
    "ST Slope: Upsloping",
    "Cholesterol Level: Borderline",
    "Cholesterol Level: High",
    "Cholesterol Level: Imputed",
    "Cholesterol Level: Normal"
]

assert len(names_cleaned)==len(names)
#%%
# Create a dictionary with list elements as keys and array elements as values
dict_lasso = {'Variables': names_cleaned, 'coefficients': Lasso.coef_[0]}
# Create DataFrame from the dictionary
summary_lasso = pd.DataFrame(dict_lasso)

summary_lasso
#%%
y_pred = Lasso.predict(X_test)
#%%
f1 = f1_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
#%%
f1
#%%
bal_acc
#%%
roc_auc = roc_auc_score(y_test, y_pred)
print(f"{roc_auc=}")
#%%
# plot roc curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, Lasso.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
# write the AUC in the plot
plt.annotate(f"AUC = {roc_auc:.2f}", (0.6, 0.4))

plt.show()

#%%
#%%
#%%
#%%
#%%

