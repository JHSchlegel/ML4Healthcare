import pandas as pd
import numpy as np
from skimpy import clean_columns
from pickle import load
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
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
#%%
df_train = pd.DataFrame(data=X_train, columns=names_cleaned)

#%%
Lasso = LogisticRegressionCV(cv=10, random_state=0).fit(df_train, y_train)


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
df_test = pd.DataFrame(data=X_test, columns=names_cleaned)

y_pred = Lasso.predict(df_test)
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
print(f"Lasso model score on training data: {Lasso.score(df_train, y_train)}")
print(f"Lasso model score on testing data: {Lasso.score(df_test, y_test)}")

#%%

#%%
#%%
coefs = pd.DataFrame(
    Lasso.coef_[0], columns=["Coefficients"], index=names_cleaned
)
coefs
#%%
coefs = pd.DataFrame({"Name": names_cleaned,"Coefficients": Lasso.coef_[0]})
coefs
# plot the coefficients of the Lasso model

import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
coefs["Coefficients"].plot(kind="bar")
plt.xticks(range(len(coefs)), coefs["Name"], rotation=90)
plt.ylabel("Coefficient")
plt.title("Coefficients of the Lasso model")
plt.show()



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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#%%
from sklearn.inspection import permutation_importance
r = permutation_importance(Lasso, df_test, y_test,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    print(f"{names_cleaned[i]:<8}"
          f"{r.importances_mean[i]:.3f}"
          f" +/- {r.importances_std[i]:.3f}")


#%%
# Calculate the upper and lower bounds for error bars
means = r.importances_mean
sds = r.importances_std
# Calculate the upper and lower bounds for error bars
upper = means + sds
lower = means - sds

# Plot the feature importances with error bars
plt.figure(figsize=(10, 6))
plt.barh(range(len(means)), means, xerr=sds, align='center', alpha=0.7, color='lightblue', ecolor='black', capsize=10)
plt.yticks(range(len(names_cleaned)), names_cleaned)
plt.xlabel('Values')
plt.title('Mean and Standard Deviation')
plt.tight_layout()
plt.show()

#%%

#%%

#%%