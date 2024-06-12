"""
Linear Regrssion - Gender Datset

1. Predict Height using Wreight
2. Predicit hoieght using weight and ismale
3. function forms for 1 and 2
bonus.contour plots

"""

__date__ = "2023-06-05"
__author__ = "Wavey Davey"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
coffee = rf'C:\Users\DavidPayne\MLE07\Probability\data\gender_cleaned_test_train.xlsx'
df1 = pd.read_excel(coffee, 'train')
df2 = pd.read_excel(coffee, 'test')

# %% --------------------------------------------------------------------------
#plot the data 
# -----------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.scatter(
    df1['Weight'], df1['Height'],
    alpha=0.3,
    c = df1['IsMale'].map({0:'orange', 1:'green'})
)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
X_train = pd.DataFrame(df1[['Weight']])
y_train = pd.DataFrame(df1['Height'])
X_test = pd.DataFrame(df2[['Weight']])
y_test = pd.DataFrame(df2['Height'])
X_train2 = pd.DataFrame(df1[['Weight','IsMale']])
X_test2 = pd.DataFrame(df2[['Weight','IsMale']])
# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
model1 = LinearRegression()
model1.fit(X_train, y_train)

model2 = LinearRegression()
model2.fit(X_train2, y_train)
#Prediction Function 
#y_hat = b + beta1*1
#predicte_height = model1.interpet_ +model1coeff_[0]*wieght

#evaluate on the test set
y_pred1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
print(f"mse model 1 : {mse1:.4f}")

y_pred2 = model2.predict(X_test2)
mse2 = mean_squared_error(y_test, y_pred2)
print(f"mse model 2 : {mse2:.4f}")


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
x = np.linspace(0,120,10)
y = model2.intercept_+ model2.coef_[[0],[0]]*x
y_1 = model2.intercept_ + model2.coef_[[0],[0]]*x + model2.coef_[[0],[1]]

fig, ax = plt.subplots()
ax.scatter(
    df1['Weight'], df1['Height'],
    alpha=0.3,
    c = df1['IsMale'].map({0:'orange', 1:'green'})
)
ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.plot(x, y, label = 'model 2 (with Ismale =1)', c='blue')
ax.plot(x, y_1, label = 'model 2 (with Ismale =0)', c='red')
ax.grid()
_ = ax.legend()


# %% --------------------------------------------------------------------------
# Bonus: contour plot
# contour plots are used to represent a 3 dimentaionals on a 3d plot 
# -----------------------------------------------------------------------------
#moxdel 1: predicted heioght= 123.05 + 0.62*weifht
# x - axis : b's
# y-axis : beta1's
#z_axis = mse
#ax.contour

def emp_risk(b, beta):
    mse = mean_squared_error(df1['Height'], b + beta*df1['Weight'])
    return mse
bs = np.linspace(-200, 600, 50)
betas = np.linspace(-5, 5, 50)


# %%
Bs, BETAs = np.meshgrid(bs, betas, indexing='ij')
# %%
EMP_RISKS = np.zeros((50, 50))
for i, b in enumerate(bs):
    for j, beta in enumerate(betas):
        EMP_RISKS[i,j] = emp_risk(b, beta)

fig, ax = plt.subplots()
levels = ax.contour(Bs, BETAs, EMP_RISKS)
fig.colorbar(levels)

# %%
fig, ax = plt.subplots()
levels = ax.contourf(X, Y, Z, levels=50)
fig.colorbar(levels)
# %%
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_surface(X, Y, Z)
# %%
