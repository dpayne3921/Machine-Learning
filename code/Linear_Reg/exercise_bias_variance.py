"""
Excericese for bias an varaince

Enter short description of the script
"""

__date__ = "2023-06-06"
__author__ = "David Payne"

#test

# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# make x and y
# -----------------------------------------------------------------------------
sample_size =15
X = rng.uniform(1, 10 , sample_size)
e = rng.uniform(0,1)
y = []
for i in X:
    y.append(math.log(i)+e)
df = pd.DataFrame({'X':X, 'y':y})


# %% --------------------------------------------------------------------------
# explore
# -----------------------------------------------------------------------------
fig, ax=plt.subplots()
ax.scatter(df['X'], df['y'])
ax.set_xlabel('X')
ax.set_ylabel('y')
_ = ax.set_title('X vs y')
print(f"{df[['X','y']].corr()}")


# %% --------------------------------------------------------------------------
# train and test 
# -----------------------------------------------------------------------------
X = df[['X']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=rng)
print(f'{X_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_train.shape = }')
print(f'{y_test.shape = }')


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'{y_pred = }')
rmse = mean_squared_error(y_test, y_pred, squared=False)
print (f'Root MSE: {rmse:.4f}')

fig, ax = plt.subplots()

ax.scatter(X_test,y_test)

ax.plot(X_test, y_pred,color='k',label='predicted')


# %%
error_array = []
y0 = []
y0hat = []

for i in range(1000):
    X = rng.uniform(1, 10 , sample_size)
    e = rng.uniform(0,1)
    y = []
    for i in X:
        y.append(math.log(i))
    df = pd.DataFrame({'X':X, 'y':y})

    X = df[['X']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=rng)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    y0.append(np.array(y_test)[5])
    y0hat.append(y_pred[5])

    error_array.append(rmse)

df1 = pd.DataFrame({'y0':y0, 'y0hat':y0hat})   


# %%
