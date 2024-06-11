#%%
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
 #%%
rng =np.random.RandomState(123)

#%%
df = pd.DataFrame(

    ['L','L','M','H','M','L','H'], columns=['Levels']

)

display(df)

#%%
#not suitable to be used in the context of ML modelling but fine for EDA
mapping = {'L':0, 'M': 1, 'H':2}
df['Levels_ord'] = df['Levels'].map(mapping)
display(df)

#2. using Scikit -Learn
model =OrdinalEncoder(categories=[['L','M','H']])

model.fit(df[['Levels']]) # remember X has to be 2 dimensional
df['Levels_ord_sklearn'] = model.transform(df[['Levels']])
display(df)

# %%
#one hot-encoding
cities = pd.DataFrame(

    {
        "City": [
            "London",
            "Belfast",
            "Manchester",
            "Liverpool",
            "London",
            "Liverpool",
            "Belfast",
        ]
    }
)
display(cities)
cities1 = pd.get_dummies(cities)
display(cities1)

#2. one hot (with m-1) withs pandas
cities2 = pd.get_dummies(cities, drop_first=True)
display(cities2)

#one hot encoding with scikit learn
model = OneHotEncoder()
model.fit(cities)
cities3 = model.transform(cities)
display(cities3)
# the output of ONEhotencoder transformation will be a sparse matrix by default
df_cities = pd.DataFrame(cities3.toarray(), columns=model.get_feature_names_out())
#one( with m-1 coulms with scikeit learn
model = OneHotEncoder(drop='first')
model.fit(cities)
cities4 = model.transform(cities)
# %%
