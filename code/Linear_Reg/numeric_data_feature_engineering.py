
# %% --------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import binarize, Binarizer
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import minmax_scale, MinMaxScaler


# %% --------------------------------------------------------------------------
#set random state 
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Generate some data 
# -----------------------------------------------------------------------------
my_array = rng.randint(255, size = 10)
my_series = pd.Series(my_array)
#%%
#first method:manual calculation
threshold =127
my_array_bin = [1 if x > threshold else 0 for x in my_array]

#second method using the helper function binarizse
#sutiable for EDA but not for machine learning pipeline
my_array_bin2= binarize(my_array.reshape(-1,1), threshold=threshold)
print(my_array_bin2)

#Third method: usin the Binarizer Class (recommened)
#suitable for piplines and working alongside ML models
bnrz = Binarizer(threshold=threshold)
bnrz.fit(my_array.reshape(-1,1))
my_array_bin3 = bnrz.transform(my_array.reshape(-1,1))
print(f'{my_array_bin3 = }')

# %%
#standard scaling
my_array_scale = (my_array - my_array.mean())/my_array.std()
print(my_array_scale)

#Second method:
#using the helper array scale
my_array_scale2 = scale(my_array.reshape(-1, 1))
print(my_array_scale2)

#Third method approach: using Standard Scaler clasee
model = StandardScaler()
model.fit(my_array.reshape(-1,1))
my_array_scale3 = model.transform(my_array.reshape(-1,1))
print(my_array_scale3)

# %%
#min-max scaling

