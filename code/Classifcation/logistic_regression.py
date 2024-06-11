"""
logistic Regression gender dataset

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


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
file_path = r'../data/gender_cleaned_test_train.xlsx'
df_train =pd.read_excel(file_path, sheet_name='train')
df_test =pd.read_excel(file_path, sheet_name='test')


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
X_train = df_train[['Height']]
X_test =df_test[['Height']]
y_train = df_train['IsMale']
y_test = df_test['IsMale']

#create model object
model = LogisticRegression()
#fit model on training data
model.fit(X_train, y_train)

#functional form of this model
# y_hat = sigmoid(b + beta_1*x)
#Predicted probability of male = sigmoid(b + beta_1 * height)
b = model.intercept_[0]
beta_1 = model.coef_[0][0]


def sigmoid(x):
    return 1/(1+np.exp(-x))

# make predictionns
y_pred = model.predict(X_test)

#calculate accuracy
print(f"Accuracy :{accuracy_score(y_test, y_pred):.4f}")

#Confusion Matrix
fig, ax = plt.subplots()
#you can plot confusion matrix usiong 'from predictions' or 'from estimator'
_ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='OrRd')
# _ = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='OrRd') 

#plotting the prediction function
x = np.linspace(140, 200, 100)

#predict proba returns probabilities for both calses so we index 2nd collum
#y = model.predict_proba(x.reshape(-1,1))[:,1]


y = sigmoid(model.intercept_ + model.coef_[0][0]*x)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.plot(x,y,c='red')
ax.grid()
ax.set_xlabel('height')
ax.set_ylabel('ismale')

# %%
