"""
Getting experience with time series data and linear regression
"""

__date__ = "12/06/2024"
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

from linreg_mods import check_sequential_dates, check_missing_values
# %% --------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
file_path = r"C:\Users\Dpayn\MLE\Machine-Learning\data\daily_20240612T1255.csv"
df = pd.read_csv(file_path, skiprows=1, engine='python')

# %% --------------------------------------------------------------------------
# Data processsing
# -----------------------------------------------------------------------------
df = df[df.columns.drop(list(df.filter(regex='SYM')))]
df = df.drop(' ID', axis=1)

# %% --------------------------------------------------------------------------
# Data processsing 2
# -----------------------------------------------------------------------------
# Melt the DataFrame to bring all the month columns into a single column
df_melted = df.melt(id_vars=['YEAR', 'DD', 'PARAM', 'TYPE'], var_name='month', value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Convert the 'month' column to its corresponding numeric value
df_melted['month'] = pd.to_datetime(df_melted['month'], format='%b').dt.month

# Convert 'DD' to integer
df_melted['DD'] = df_melted['DD'].astype(int)

# Convert 'DD' to two-digit string
df_melted['DD'] = df_melted['DD'].apply(lambda x: f'{x:02d}')

# Combine 'YEAR', 'month', and 'DD' to create a datetime column
df_melted['date'] = pd.to_datetime(df_melted['YEAR'].astype(str) + df_melted['month'].astype(str) + df_melted['DD'], format='%Y%m%d', errors='coerce')

df_melted = df_melted.drop(['YEAR', 'month', 'DD'], axis=1)


# %%
# Pivot the DataFrame to bring each unique 'PARAM' value into a new column
df_pivoted = df_melted.pivot_table(index=['date'], columns='PARAM', values='value', aggfunc='mean').reset_index()

# Flatten the columns
df_pivoted.columns = df_pivoted.columns.get_level_values(0)

# Reset the index
df_pivoted.reset_index(drop=True, inplace=True)


# %%

# Create a date range from 1914-05-23 to the latest date in your DataFrame
date_range = pd.date_range(start=df_pivoted['date'].min(), end=df_pivoted['date'].max())
# Convert the 'date' column to datetime if it's not already
df_pivoted['date'] = pd.to_datetime(df_pivoted['date'])

# Set the 'date' column as the index
df_pivoted.set_index('date', inplace=True)

# Reindex the DataFrame with the date range
df_pivoted = df_pivoted.reindex(date_range)

# Reset the index
df_pivoted.reset_index(inplace=True)
df_pivoted.rename(columns={'index': 'date'}, inplace=True)
 
# %%
import matplotlib.pyplot as plt

# Filter the DataFrame for years from 1980 to 2019
df_filtered = df_pivoted[(df_pivoted['date'].dt.year >= 1980) & (df_pivoted['date'].dt.year <= 2019)]

fig, axs = plt.subplots(8, 1, figsize=(10, 60))
#jkj
for i, year in enumerate(range(1980, 2020, 5)):
    df_period = df_filtered[(df_filtered['date'].dt.year >= year) & (df_filtered['date'].dt.year < year + 5)]
    axs[i].plot(df_period['date'], df_period[1])
    axs[i].set_title(f'Years: {year} - {year + 4}')
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('1')

plt.tight_layout()
plt.show()
#tu

# %% --------------------------------------------------------------------------
# test data
# -----------------------------------------------------------------------------

# Extract the date column and convert it to a list
dates = df_filtered['date'].tolist()

# Call the function to check if dates are sequential
is_valid, message = check_sequential_dates(dates)
print(message)

# Assuming df_filtered is your DataFrame and it is already loaded

missing_values = check_missing_values(df_filtered)
print(missing_values)

# %%
