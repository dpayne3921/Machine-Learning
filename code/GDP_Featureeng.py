"""
Regression deep divein python

simple linear regression on GDP vs Happniess datadset

"""

__date__ = "2022-05-24"
__author__ = "WaveyDavey"



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
# load data
# -----------------------------------------------------------------------------
file_path = r'../data/gdp-vs-happiness.csv'
df = pd.read_csv(file_path)

# %% --------------------------------------------------------------------------
# data pre-processing 
# -----------------------------------------------------------------------------
#rename gdp and life satification columns
df.rename(
    columns={df.columns[3]:'GDPPC', df.columns[4]: 'life satifcation'},
    inplace=True)
#drop unamed coulmn
df.drop(columns = df.columns[5], inplace =True)
# Pick the rows only for the year 2016

df = df[df['Year'] == 2016]
df.dropna(inplace=True)
#create a new column log(GDPPC)
df['LogGDPPC'] = np.log(df['GDPPC'])
display(df)


# %% --------------------------------------------------------------------------
# exploring Data Analysis
# 
# -----------------------------------------------------------------------------
#plot scatterplot of GDPPC vs Life Satification
fig, ax = plt.subplots()
ax.scatter(df['LogGDPPC'], df['life satifcation'])
ax.set_xlabel('GDP per capita')
ax.set_ylabel('life Satisfaction')
_ = ax.set_title('GDP per capita vs life satifcation (2016)')
print(f"{df[['LogGDPPC','life satifcation']].corr()}")

# %%
X = df[['LogGDPPC']] 
y = df['life satifcation']
#split the data into train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=rng
)

print(f'{X_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_train.shape = }')
print(f'{y_test.shape = }')
#print the shape of X_train
#You will often see the followiung error while fitting the model:
#reshape ypur data either using array.reshape(-1, 1) if you data has a ignle feature
#or
#array.reshape(1,-1) if it contains a single sample
# this happen when X is defined as 1D object
#in the case of a numpy array, you can do array.reshape(-1,1)
#in the case of a pandas object, you can create a DataFrame instead of a series
#for X_train and X_trest by using double square brackets on the feature name

# %% --------------------------------------------------------------------------
# crea the model object 
# -----------------------------------------------------------------------------

model = LinearRegression()


# %% --------------------------------------------------------------------------
#train 
# -----------------------------------------------------------------------------
model.fit(X_train, y_train)
#this will learn parameters beto 0 and beta 1
#beta 0 accesed using model.intercept
#beta 1 accessed using model.coeff

#thre model is learning predificted life satisfaction which is given y:
#beta 0 + beta 1 * GDPPC

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
print(f'{y_pred = }')


# %% --------------------------------------------------------------------------
#evelue 
# -----------------------------------------------------------------------------
rmse = mean_squared_error(y_test, y_pred, squared=False)
print (f'Root MSE: {rmse:.4f}')


# %% --------------------------------------------------------------------------
# Exercise :plot a scatterplot  of the test set
# and plot the line fit by linear regression ( Bo + B1*GDPPC)
# -----------------------------------------------------------------------------

# %%

fig, ax = plt.subplots()

ax.scatter(X_test,y_test)

ax.plot(X_test, y_pred,color='k',label='predicted')
# %%