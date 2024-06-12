"""
Regularization

Prostate Dataset
"""

__date__ = "2022-01-13"
__author__ = "Wavey Davey"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
file_path = r"../data/prostate/prostate.data"
df = pd.read_csv(file_path, sep="\t", index_col=0)
# Separate data into training and test sets
df_train = df[df["train"] == "T"]
df_test = df[df["train"] == "F"]


# %% --------------------------------------------------------------------------
# # Create X and y for train and test sets
# -----------------------------------------------------------------------------
X_train = df_train.drop(columns=["lpsa", "train"])
y_train = df_train["lpsa"]
X_test = df_test.drop(columns=["lpsa", "train"])
y_test = df_test["lpsa"]


# %% --------------------------------------------------------------------------
# Scale the data
# -----------------------------------------------------------------------------
scaler = StandardScaler()
Xt_train = scaler.fit_transform(X_train)
Xt_test = scaler.transform(X_test)  # Transforming the test set


# %% --------------------------------------------------------------------------
# OLS
# -----------------------------------------------------------------------------
model = LinearRegression()
model.fit(Xt_train, y_train)
mse_train = mean_squared_error(y_train, model.predict(Xt_train))
print(mse_train)


# %% --------------------------------------------------------------------------
# Ridge Regression
# -----------------------------------------------------------------------------
results = {}  # store mse
coefs = {}  # store coefs
for alpha in np.linspace(0, 100, 100):
    model = Ridge(alpha=alpha)
    model.fit(Xt_train, y_train)
    coefs[alpha] = dict(zip(list(X_train.columns), model.coef_))

    y_pred = model.predict(Xt_test)
    mse = mean_squared_error(y_test, y_pred)
    results[alpha] = mse

# Plot mse vs regularization strength (alpha/lambda)
fig, ax = plt.subplots()
ax.plot(list(results.keys()), list(results.values()))
ax.set_xlabel("alpha/lambda")
ax.set_ylabel("mse")
ax.set_title("Ridge Regression - MSE vs Alpha/Lambda")

# Plot coefficients vs regularization strength (alpha/lambda)
fig, ax = plt.subplots(figsize=(12, 8))
df_params = pd.DataFrame(coefs).T
df_params.plot(ax=ax)
ax.set_ylim(-1, 1)
ax.axhline(xmax = 100)
ax.set_xlabel("alpha/lambda")
ax.set_ylabel("coefficient")
_ = ax.set_title("Ridge Regression - Coefficients vs Alpha/Lambda")

# %% --------------------------------------------------------------------------
# Lasso Regression
# -----------------------------------------------------------------------------
results = {}  # store mse
coefs = {}  # store coefs
for alpha in np.linspace(0.01, 1, 100):
    model = Lasso(alpha=alpha)
    model.fit(Xt_train, y_train)
    coefs[alpha] = dict(zip(list(X_train.columns), model.coef_))

    y_pred = model.predict(Xt_test)
    mse = mean_squared_error(y_test, y_pred)
    results[alpha] = mse

# Plot mse vs regularization strength (alpha/lambda)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(list(results.keys()), list(results.values()))
ax.set_xlabel("alpha/lambda")
ax.set_ylabel("mse")
ax.set_title("Lasso Regression - MSE vs Alpha/Lambda")

# Plot coefficients vs regularization strength (alpha/lambda)
fig, ax = plt.subplots(figsize=(12, 8))
df_params = pd.DataFrame(coefs).T
df_params.plot(ax=ax)
ax.set_ylim(-1, 1)
ax.axhline(xmax = 100)
ax.axvline(min(results, key=lambda k: results[k]))
ax.set_xlabel("alpha/lambda")
ax.set_ylabel("coefficient")
ax.legend(loc="right")
_ = ax.set_title("Lasso Regression - Coefficients vs Alpha/Lambda")

# %%
