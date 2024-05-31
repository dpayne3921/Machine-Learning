#%%
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Dpayn\MLE\Machine-Learning\Large_data\water_model1.h5')

# Function to clean numeric columns
def clean_numeric(column):
    # Remove any non-numeric characters and convert to float
    column = column.replace({r'\*': '', ' ': '', ',': ''}, regex=True)
    return pd.to_numeric(column, errors='coerce')

# Function to preprocess data
def preprocess_data(input_data):
    # Drop non-numeric columns
    input_numeric = input_data.select_dtypes(include=['number'])

    # Clean numeric columns
    input_numeric_cleaned = input_numeric.apply(clean_numeric)

    # Fill NaN values with zeros (you can adjust this according to your data)
    input_numeric_cleaned.fillna(0, inplace=True)

    # Normalize numeric data
    scaler = MinMaxScaler()
    input_numeric_scaled = scaler.fit_transform(input_numeric_cleaned)

    return input_numeric_scaled

# Streamlit interface
st.title('Water Quality Prediction')
st.write('Adjust the feature values to predict the pH level.')

# Create input fields for each feature
input_data = {}
input_data_columns = ['BOD_MGL', 'DO_MGL', 'NO3_N_MGL', 'NO2_N_MGL', 'NH4_N_MGL', 'P_SOL_MGL', 'tmax', 'tmin', 'af', 'rain']
for column in input_data_columns:
    input_data[column] = st.slider(column, min_value=0.0, max_value=10.0, step=0.1)

# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data])

# Preprocess input data
input_data_preprocessed = preprocess_data(input_data_df)

# Reshape input data to match model's input shape
input_data_reshaped = np.reshape(input_data_preprocessed, (input_data_preprocessed.shape[0], 1, input_data_preprocessed.shape[1]))

# Make predictions
prediction = model.predict(input_data_reshaped)
predicted_ph = prediction[0][0]

# Display the prediction
st.write(f'Predicted pH level: {predicted_ph}')
# %%
