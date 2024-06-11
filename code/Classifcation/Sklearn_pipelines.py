"""
Scikit_learn pipelines Demo

usses KNN on IRis dataset with satandar scalar
"""

__date__ = "2023-08-02"
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

from sklearn.pipeline import Pipeline


# %% --------------------------------------------------------------------------
#randoms state 
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
#Load data and train tes 
# -----------------------------------------------------------------------------
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test =train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=rng
)

# %% --------------------------------------------------------------------------
#create pipeline 
# -----------------------------------------------------------------------------
#first argument is a list of tuples containing the stages
#each tuple contains two objects
#--FIST STAGE
#first element in tuple is a name refering to thte first stage
#Second element is the acatual estimator object
#--SECOND STAGE
#call the classifier
pipeline = Pipeline(
    [
    ('std_scale', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
    ]
)


# %% --------------------------------------------------------------------------
#fit and predicit workflow on the pipeline 
# -----------------------------------------------------------------------------
#calling fit on the pipeline object (with the train data)
#this will fit and transform X_train with StandardScaler followed by
#fitting the KNeighboursclassifere on (Xt_train, y_train)
pipeline.fit(X_train, y_train)

#Calling predict on the piplein object (with test data)
#this will transform X_test with stanard callers(that was fit using Z)_Train
#followed by prediction using knewiboursclassifer(that was fit using traindata)

y_pred = pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

# %%
