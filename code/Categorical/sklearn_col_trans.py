"""
column tranforms in scikit learn

this demo usess a synthesic, toy datset
"""

__date__ = "2023-08-02"
__author__ = "davidpayne"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)
 
 # %% --------------------------------------------------------------------------
 #Generate some data 
 # -----------------------------------------------------------------------------
#5 variables (columns) : 3 numerical and 2 categotaraical
#  10 samples (rows)

#genrate numerica data
num1 =rng.normal(170, 8, 10).round(2)
num2 = rng.normal(0, 1, 10).round(2)
num3 = rng.normal(70000, 25000, 10).astype(int)

#Generate categorical data
cat1 = rng.randint(4, size=10)
cat1_levels = {0: 'London', 1: 'Liverpool', 2: 'Manchester', 3: 'Belfast'}

cat2 = rng.randint(2, size=10)
cat2_levels = {0: 'Single', 1: 'Married'}
# Put everything into a dataframe
df = pd.DataFrame(
    {
        'a': num1,
        'b': num2,
        'c': cat1,
        'd': num3,
        'e': cat2
    }
)
df['c'] = df['c'].map(cat1_levels)
df['e'] = df['e'].map(cat2_levels)
display(df)


# %% --------------------------------------------------------------------------
#column transformations
# for columns 'a' and d'd we apply StandardScaler()
# for coulumns 'c' and 'e' we appply one hotencoder()
# for column ' b' , we dont apply any transformation 
# -----------------------------------------------------------------------------
#create a columntransformer object
# The first argument is a list of tuples. Each Tuple contains three elements
#The first element in the tuple is a name referrring to the transofmration
# The second element is the tranformer model object
# thrid element is the list of columns on  which the transformation is applied
# you can have multiple objects of the same type
# (example, Binaraizers with different thresholds for different columns)
# Another argument that is useful is the remainder argument
# remainder = 'drop' (default) will drop the remaining columns from the transformed output
# remainder ='passthrough' will add the remaining columns without any tranformation



col_trans = ColumnTransformer(
    [
        ("std_scale", StandardScaler(), ['a', 'd']),
        ("ohe", OneHotEncoder(), ['c']),
        ("ohe2", OneHotEncoder(drop='first'), ['e'])
    ],
    remainder='passthrough'
)

#fit the tranformer
col_trans.fit(df)

#Transform the data
df_T = col_trans.transform(df)

df_T = pd.DataFrame(df_T, columns=col_trans.get_feature_names_out())
display(df_T)
#put the data into a dataframe

# %%
