"""
perfomance evalu metric for classid
"""

__date__ = "2023-06-05"
__author__ = "David Payne"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# fetch data
# -----------------------------------------------------------------------------
y_test = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])

y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])


# %% --------------------------------------------------------------------------
# confusion matric 
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print(cm)

cm_plot = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# %% --------------------------------------------------------------------------
# precision 
# -----------------------------------------------------------------------------
pre = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"precision: {pre:.4f}")
print(f"acc: {acc:.4f}")
print(f"rec: {rec:.4f}")

# %% --------------------------------------------------------------------------
# roc_curve
# -----------------------------------------------------------------------------
y_test = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
y_pred_proba = np.array([0.05, 0.17, 0.25, 0.32, 0.41, 0.49, 0.55, 0.67, 0.81, 0.95])
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = roc_auc_score(y_test, y_pred_proba)

roc_plot = RocCurveDisplay.from_predictions(y_test,y_pred_proba)
# %%
