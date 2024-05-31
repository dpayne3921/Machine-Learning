"""
Examining ph qual in water sights

For stormharvester app
"""

__date__ = "30-05-24"
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
import geopandas as gpd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthBegin

# %% --------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

# Define the file path
file_path1 = r'C:\Users\Dpayn\MLE\Machine-Learning\Large_data\River_Water_Quality_Monitoring_1990_to_2018.csv'
file_path2 = r'C:\Users\Dpayn\MLE\Machine-Learning\Large_data\Armagh Weather.csv'  
# Load the CSV file
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
# Print the DataFrame
print(df2)


# %% --------------------------------------------------------------------------
# Process the initial dataset
# -----------------------------------------------------------------------------
 
# Define the columns to check
columns_to_check = ['ALK_MGL', 'BOD_MGL', 'COND_USCM', 'CUSOL1_MGL', 'CUSOL2_UGL', 'DO_MGL', 'FESOL1_UGL', 'NO3_N_MGL', 'NO2_N_MGL', 'NH4_N_MGL', 'PH', 'P_SOL_MGL', 'SS_MGL', 'ZN_SOL_UGL']

# Calculate the percentage of non-missing values in each column
non_missing_percentages = df1[columns_to_check].notna().mean() * 100

# Print the percentages
# print(non_missing_percentages)

# Drop any column which has less than 90% non-missing values
df1 = df1.drop(columns=[col for col in columns_to_check if non_missing_percentages[col] < 90])

# Define the additional columns to drop
additional_columns_to_drop = ['OBJECTID', 'RWB_ID_RBP2', 'Time', 'Depth']

# Drop the additional columns that exist in df1
df1 = df1.drop(columns=[col for col in additional_columns_to_drop if col in df1.columns])

# Convert 'Date' to datetime and only keep the date
df1['Date'] = pd.to_datetime(df1['Date']).dt.date
df1['Date'] = pd.to_datetime(df1['Date'])


# Print the DataFrame
# print(df1)

# %% --------------------------------------------------------------------------
# Process the second dataset dataset
# -----------------------------------------------------------------------------
# Drop rows with NaNs in 'yyyy' or 'mm'
df2 = df2.dropna(subset=['yyyy', 'mm'])

# Combine 'yyyy' and 'mm' to create a 'Date' column
df2['Date'] = pd.to_datetime(df2['yyyy'].astype(int).astype(str) + df2['mm'].astype(int).astype(str), format='%Y%m')

# Set the day to the middle of the month
df2['Date'] = df2['Date'] + MonthBegin(1) - pd.Timedelta(days=1)

# Define the columns to drop
columns_to_drop = ['yyyy', 'mm', 'sun']

# Drop the columns that exist in df2
df2 = df2.drop(columns=[col for col in columns_to_drop if col in df2.columns])

# Print the DataFrame
#print(df2)

# %% --------------------------------------------------------------------------
# Merge the two datasets
# -----------------------------------------------------------------------------

# Define a function to find the closest date in df2 for each date in df1
def find_closest_date(row):
    closest_date_index = (df2['Date'] - row['Date']).abs().idxmin()
    return df2.loc[closest_date_index, ['tmax', 'tmin', 'af', 'rain']]

# Apply the function to df1
df1[['tmax', 'tmin', 'af', 'rain']] = df1.apply(find_closest_date, axis=1)

# %% --------------------------------------------------------------------------
# Save dataset
# -----------------------------------------------------------------------------
# Define the file path
file_path = r"C:\Users\Dpayn\MLE\Machine-Learning\Large_data\waterq_data.csv"

# Save df1 to a CSV file
df1.to_csv(file_path, index=False)

# %% --------------------------------------------------------------------------
# Merge the two datasets 2
# ------------------------
# Load df1 from the CSV file, parsing the 'Date' column as a datetime
df1 = pd.read_csv(file_path, parse_dates=['Date'])

# List of columns to interpolate
columns_to_interpolate = ['BOD_MGL', 'DO_MGL', 'NO3_N_MGL', 'NO2_N_MGL', 'NH4_N_MGL', 'PH', 'P_SOL_MGL']

# Identify sites where all records across all columns to interpolate are NaN
sites_to_remove = df1.groupby('Site_Code').filter(lambda x: x[columns_to_interpolate].isna().all().all())['Site_Code'].unique()

# Remove these sites from df1
df1 = df1[~df1['Site_Code'].isin(sites_to_remove)]

# Now perform the interpolation as before
df1[columns_to_interpolate] = df1.groupby(['Site_Code', 'Date'])[columns_to_interpolate].transform(lambda group: group.interpolate(method='linear').bfill().ffill())


# Fill missing values with the previous value in the same site
df1[columns_to_interpolate] = df1.groupby('Site_Code')[columns_to_interpolate].ffill()

# Backfill any remaining missing values
df1[columns_to_interpolate] = df1.groupby('Site_Code')[columns_to_interpolate].bfill()

# Drop rows with missing values
df1 = df1.dropna()

# Print the DataFrame
print(df1.info())

# %% --------------------------------------------------------------------------
# Save dataset Cleaned
# -----------------------------------------------------------------------------
# Define the output file path
output_file_path = r"C:\Users\Dpayn\MLE\Machine-Learning\Large_data\Cleaned_water1.csv"

# Save the DataFrame to a CSV file
df1.to_csv(output_file_path, index=False)
# %% ------------------------------------------------------------
# Define the file path
#--------------------------------------------------------
file_path = r"C:\Users\Dpayn\MLE\Machine-Learning\Large_data\River_Water_Quality_Monitoring_1990_to_2018_-_pH.geojson"

# Load the GeoJSON file
gdf = gpd.read_file(file_path)

# Plot the GeoDataFrame
gdf.plot()

# Show the plot
plt.show()
# %%
