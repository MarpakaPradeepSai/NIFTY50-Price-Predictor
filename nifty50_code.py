import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the trained model
model_filename = 'LSTM_NIFTY_10(169.54).h5'  # Use your model's filename here
try:
    model = load_model(model_filename)
    st.success("LSTM model loaded successfully.")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")

# Function to create dataset
def create_dataset(data, look_back=1):
    X = []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        X.append(a)
    return np.array(X)

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-size: 49px;'>NIFTY Stock Price Predictor 📈📉💰</h1>", unsafe_allow_html=True)

# User input for past prices
input_data = st.text_area("Input Past Prices (comma-separated):", "10000, 10100, 10200, 10300, 10400, 10500")

# Fixed look back period
look_back = 10

# Button for prediction
if st.button("Predict"):
    try:
        input_data = np.array([float(i) for i in input_data.split(',')])
        input_data = input_data.reshape(-1, 1)

        # Scale the input data
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_data_scaled = scaler.fit_transform(input_data)

        # Create dataset for the model
        X_input = create_dataset(input_data_scaled, look_back)
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

        # Make prediction
        prediction = model.predict(X_input)
        prediction = scaler.inverse_transform(prediction)

        # Display the prediction
        st.subheader('Predicted Price:')
        st.write(f"${prediction[-1][0]:.2f}")

        # Plotting the input data and prediction
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(input_data)), scaler.inverse_transform(input_data_scaled), label='Input Prices', color='blue')
        plt.axhline(y=prediction[-1][0], color='red', linestyle='--', label='Predicted Price')
        plt.title('NIFTY Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        st.pyplot(plt)

    except ValueError:
        st.error("Please ensure all inputs are valid numbers.")

# Add some styling for the button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: green; /* Green color for the Predict button */
        color: white;
    }
    div.stButton > button:focus,
    div.stButton > button:hover,
    div.stButton > button:active {
        color: white; /* Keep text white on hover, focus, and active states */
        outline: 2px solid green; /* Green outline for the clicked button */
    }
    </style>
    """,
    unsafe_allow_html=True
)
