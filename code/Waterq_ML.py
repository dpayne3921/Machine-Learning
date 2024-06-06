"""
Building ML model to predict water quality

For stormharvester app
"""

__date__ = "30-05-24"
__author__ = "WaveyDavey"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# %% --------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
# Load the dataset
data = pd.read_csv(r'C:\Users\Dpayn\MLE\Machine-Learning\Large_data\Cleaned_water1.csv')

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by Site_Code and Date
data = data.sort_values(by=['Site_Code', 'Date'])


# %% --------------------------------------------------------------------------
# Clean Data
# -----------------------------------------------------------------------------

# Function to clean numeric columns
def clean_numeric(column):
    # Remove any non-numeric characters and convert to float
    column = column.replace({r'\*': '', ' ': '', ',': ''}, regex=True)
    return pd.to_numeric(column, errors='coerce')

# Apply the cleaning function to relevant columns
numeric_columns = ['BOD_MGL', 'DO_MGL', 'NO3_N_MGL', 'NO2_N_MGL', 'NH4_N_MGL', 'PH', 'P_SOL_MGL', 'tmax', 'tmin', 'af', 'rain']
for col in numeric_columns:
    data[col] = clean_numeric(data[col])

# Drop rows with NaN values resulting from coercion
data = data.dropna()

# Confirm the changes
print(data.info())
print(data.head())



#%%
#testing
import sys
for path in sys.path:
    print(path)


# %% --------------------------------------------------------------------------
#  Feature Engineering
# -----------------------------------------------------------------------------
# Create lag features for pH and other relevant columns
lag_features = ['PH', 'BOD_MGL', 'DO_MGL', 'NO3_N_MGL', 'NO2_N_MGL', 'NH4_N_MGL', 'P_SOL_MGL', 'tmax', 'tmin', 'af', 'rain']
for feature in lag_features:
    data[f'{feature}_lag1'] = data.groupby('Site_Code')[feature].shift(1)

# Drop rows with NaN values (which are the rows without lag data)
data = data.dropna()

# Encode categorical variables if necessary (example shown for 'Site_Status_21Oct2020' and 'Primary_Basin')
data = pd.get_dummies(data, columns=['Site_Status_21Oct2020', 'Primary_Basin'])

# Normalize features (excluding the target variable 'PH')
features_to_scale = data.columns.difference(['PH', 'Date', 'Site_Code', 'Station_Name', 'GlobalID'])
scaler = MinMaxScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Prepare input and output
X = data.drop(columns=['PH', 'Date', 'Site_Code', 'Station_Name', 'GlobalID'])
y = data['PH']

# Split the data into training and testing sets without shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# %% --------------------------------------------------------------------------
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = np.mean(np.abs(predictions - y_test.values))
print(f'Mean Absolute Error: {mae}')

# %%
# Define the path to save the trained model
model_path = r'C:\Users\Dpayn\MLE\Machine-Learning\Large_data\water_model1.h5'

# Save the trained model
model.save(model_path)

