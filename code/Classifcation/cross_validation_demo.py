"""
Cross validation i scikit-learn

KNN on Iris data
"""

__date__ = "2023-09-02"
__author__ = "davidpayne"



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

from sklearn.model_selection import cross_val_score, cross_validate

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
# model 1: 10-fold cross validation with cross_val_score 
# -----------------------------------------------------------------------------
knn1 = KNeighborsClassifier() #n_neighours =5
scores = cross_val_score(knn1, X_train, y_train, cv=10)
scores2= cross_validate(knn1, X_train, y_train, cv=10, scoring=['accuracy','precision_macro'])
print(scores2)
print(f"{scores.mean()*100:.2f} accuracy with a standard deviation {scores.std()*100:.2f}")
# %%
