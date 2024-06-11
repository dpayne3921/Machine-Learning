"""
Regularisation Demo

Geometric interpretation of L1 and L2
"""

__date__ = "2022-07-23"
__author__ = "RamVaradarajan"
__version__ = "0.1"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


# %% --------------------------------------------------------------------------
# Create function
# -----------------------------------------------------------------------------
def f(x, y):
    z = 10 * (x + 3) ** 2 + 100 * (y - 2) ** 2
    return z

# %% --------------------------------------------------------------------------
# Generate data
# -----------------------------------------------------------------------------
# Generate the data
x = np.arange(-5, 5, 0.02)
y = np.arange(-5, 5, 0.02)

# Create a grid of points combining values from x and y
X, Y = np.meshgrid(x, y, indexing='ij')
Z = f(X, Y)

# Create functions to plot L1 and L2 norm
def norm(p):
    return ((np.abs((X)) ** p) + (np.abs((Y)) ** p)) ** (1.0 / p)

L1 = norm(1)
L2 = norm(2)


# %% --------------------------------------------------------------------------
# Plot function and norms
# -----------------------------------------------------------------------------
# Draw the contour plot
fig, ax = plt.subplots(figsize=(14, 12))
contour_set = ax.contour(X, Y, Z, levels=np.arange(160,220,10), cmap="Set2")
fig.colorbar(contour_set)  # displays the levels in a colorbar
ax.clabel(contour_set)  # displays the value of each contour
_ = ax.set_xlim(-2, 2)
_ = ax.set_ylim(-2, 2)
_ = ax.set_xticks(np.arange(-2, 2, 1))
_ = ax.set_yticks(np.arange(-2, 2, 1))
ax.grid()

# ax.add_patch(plt.Circle((0,0),1,color='r',fill=False)) # add L2 norm
ax.contour(X, Y, L1, [1], colors="b")  # add L1 norm
ax.contour(X, Y, L2, [1], colors="black")  # add L2 norm

# %% --------------------------------------------------------------------------
# Zoom in
# -----------------------------------------------------------------------------
# Draw the contour plot
fig, ax = plt.subplots(figsize=(14, 5))
# contour_set = ax.contour(X, Y, Z, levels=[180, 181, 182, 189, 190, 191], cmap="Set1")
contour_set = ax.contour(X, Y, Z, levels=range(180,191), cmap="Set2")
fig.colorbar(contour_set)  # displays the levels in a colorbar
ax.clabel(contour_set)  # displays the value of each contour
_ = ax.set_xlim(-0.4, 0.1)#, 0.01)
_ = ax.set_ylim(0.9, 1.1)#, 0.01)
_ = ax.set_xticks(np.arange(-0.4, 0.1, 0.1))
_ = ax.set_yticks(np.arange(0.9, 1.1, 0.1))
ax.grid()

ax.contour(X, Y, L1, [1], colors="b")  # add L1 norm
ax.contour(X, Y, L2, [1], colors="black")  # add L2 norm

# %%