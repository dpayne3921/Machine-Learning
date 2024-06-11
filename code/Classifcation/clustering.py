"""
clustering example

useses the mouse dataset
"""

__date__ = "2024-11-06"
__author__ = "Wavey Davey"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
df = pd.read_csv(r'../data/mouse.csv', sep=' ', header=None)

X = df[[0, 1]]
# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(df[0], df[1])
#ax.scatter(df[0], df[1], c=df[2].map(
 #   {'Head': 'cyan',
  #   'Ear_left': 'orange',
  #   'Ear_right': 'magenta',
  #   'Noise': 'black'}
#))


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
km = KMeans(n_clusters = 3)
km.fit(X) #Note that y is not passed into unsupervised training models fit methods
y_pred = km.predict(X)


# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(df[0], df[1], c=y_pred)


# %% --------------------------------------------------------------------------
# elbow plot
# -----------------------------------------------------------------------------
km_sse = []
for k in range (1, 19):
    print(f'Testing for {k= }')
    km = KMeans(n_clusters=k)
    km.fit(X)
    km_sse.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(np.arange(1, 19), km_sse)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')

# %%
