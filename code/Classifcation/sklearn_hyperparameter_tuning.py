"""
GridSearch CV and Randomized Seach CV


"""

__date__ = "2023-09-02"
__author__ = "davidpayne"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
rng=np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test =train_test_split(
    iris.data, iris.target, test_size =0.2, random_state=rng
)

# %% --------------------------------------------------------------------------
#Grid Search with 10-fold CV 
# -----------------------------------------------------------------------------
knn1 = KNeighborsClassifier()

# Create a dictionary for our parameter grid
params = {
    'n_neighbors':[3,5,7,9,11],
    'p':[1,2]
}

#Create the GridSearchCV object
gscv = GridSearchCV(knn1, params, cv =10)

# Fit the model
gscv.fit(X_train, y_train)

#you can get the best model by using gscv.best_estimator_
best_model = gscv.best_estimator_

# you can also get the best set of hyperparmenters by using gscv.best_params_
best_hyperparams = gscv.best_params_

#ALL the best cross validation results are stored in gscv.cv_results_
# %%
