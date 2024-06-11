"""
KNN on Irish

with and withoutstandard scalar
"""

__date__ = "2023-06-08"
__author__ = "Wavey Davey"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test =train_test_split(
    iris.data, iris.target, test_size =0.2, random_state=rng
)

# %% --------------------------------------------------------------------------
# model 1: without standardization
# -----------------------------------------------------------------------------
knn1 = KNeighborsClassifier()
knn1.fit(X_train, y_train)
y_pred1 = knn1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
print(f"Accuracy (withouth standarisation : {acc1:.4f})")

# %% --------------------------------------------------------------------------
# model 2 with standardisation
# -----------------------------------------------------------------------------
std_scaler = StandardScaler()
#calling .fit will learn the transformation model paremeters
#In this case, the mean and standard deviation from the train data
std_scaler.fit(X_train)

#calling .transform on X_train will produce Xt_train
Xt_train = std_scaler.transform(X_train)

#The above two steps can be replaced with fit_transform method
#Xt_train = std_scaler.fit_transform(X_train)

Xt_test = std_scaler.transform(X_test)

#now, we can do the fit and predict workflow on the transformed data
knn2 = KNeighborsClassifier()
knn2.fit(Xt_train, y_train)
y_pred2 = knn2.predict(Xt_test)
acc2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy (without standardisation): {acc2:4f}")
# %%



# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
ohe = One