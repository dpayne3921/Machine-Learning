"""
classifcation deep dive in pytohn

a demo to illustrate kNN claiffifer on the MNIST dataset
"""

__date__ = "2023-06-05"
__author__ = "David Payne"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import colors
from sklearn.neighbors import NearestNeighbors
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# fetch data
# -----------------------------------------------------------------------------
mnist = fetch_openml(data_id=554, as_frame=False)

# %%
#for the purpose of this demoexample, we are going to work with a small
#subset of the data
#shuffle and get a subset of 10000 samples fro the datea
mnist.data, mnist.target =shuffle(mnist.data, mnist.target, random_state=rng)

#take the first 10000 samples from the shuffled arrays
X = mnist.data[:10000]
y = mnist.target[:10000]

#print the shape of X and y
print(f'{X.shape = }')
print(f'{y.shape = }')

# %% --------------------------
#prepare the data (Train/test split)
#10% of the samples in the test set
#print out the shapes for X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rng)

print(f'{X_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_train.shape = }')
print(f'{y_test.shape = }')
# %% --------------------------
#create the model object
model = KNeighborsClassifier()

# %%
#train: fitting the model to train data
#in this phase, we fit the model using the train inputs and outputs
model.fit(X_train, y_train)

# %%
# test: make predictions on the test data
#in this phase, we make predictions with the @trained' model om the test inputs
#note: you dont pass in y_test into the predict method
#this returns the predicted values( y_pred) for the test inputs (X_test) 
y_pred = model.predict(X_test)
print(f'{y_pred.shape =}') # This should be the same as y_test.shape


# %% --------------------------------------------------------------------------
# Evaluation phase: compare the model performance
# -----------------------------------------------------------------------------
#in this phase we compare the predictied outputs and actual outputs
acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc_score}')

# %%
#plt.figure(figsize=(5, 5))
#plt.subplots_adjust(hspace=0.5)
items = [0,1,2,3,4,5,6,7,8]

for n, item in enumerate(items):
    # add a new subplot iteratively
    ax = plt.subplot(3, 3, n + 1)

    # filter df and plot ticker on the new subplot axis

    new_shape = np.reshape(X_test[item],(28,28) )
    #cmap = colors.ListedColormap(['green', 'yellow'])
    #bounds = [0,125,256]
    #norm = colors.BoundaryNorm(bounds, cmap.N)

    #fig, ax = plt.subplots()
    ax.imshow(new_shape, cmap='gray')#, norm=norm)
    ax.set_title(f'True = {y_test[item]}, Pred= {y_pred[item]} ')
plt.subplots_adjust(
                    wspace=0.9,
                    hspace=0.9)
plt.show()
# %%
#Fit the k-nearest neighbors model

n_closest = 5
x=51
ind =model.kneighbors(
    X=X_test[x, :].reshape((1,-1)), n_neighbors= n_closest, return_distance=False

)
fig, ax =plt.subplots(nrows=1, ncols= n_closest+2, figsize =(7,1))
img = X_test[x,:].reshape((28,28))
ax.flat[0].imshow(img, cmap =plt.cm.gray)
ax.flat[0].axis('off')
ax.flat[0].set_title(f'predicted {y_pred[x]}')
ax.flat[1].axis('off')
for i in range(0, n_closest):
    img = X_train[ind[0][i], :].reshape((28,28))
    ax.flat[i + 2].imshow(img, cmap=plt.cm.gray)
    ax.flat[i + 2].axis('off')
plt.show


# %% --------------------------------------------------------------------------
# Confusion matrix
# -----------------------------------------------------------------------------

# %%
