import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the LSTM model
model = load_model("LSTM_NIFTY_10(169.54).h5")

# Load and prepare your data
def load_data():
    # Replace with your actual data loading logic
    df = pd.read_csv('your_data.csv', parse_dates=['Date'], index_col='Date')
    return df

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Predict function
def predict_price(data, look_back=10):
    scaled_data, scaler = scale_data(data)
    X, _ = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)

# Streamlit app layout
st.title("NIFTY Stock Price Prediction")

# Load data
df = load_data()
st.write("Data Loaded Successfully")

# Fixed look-back period
look_back = 10

# Show data statistics
st.write(df.describe())

# Allow users to input a date range
start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

# Filter data based on the selected date range
if start_date and end_date:
    filtered_data = df.loc[start_date:end_date]['Close'].values.reshape(-1, 1)
    
    if len(filtered_data) > look_back:
        predictions = predict_price(filtered_data, look_back)
        
        # Plotting the results
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_data[look_back:], label='Actual Price')
        plt.plot(predictions, label='Predicted Price', linestyle='--')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Not enough data points to predict. Please select a different date range.")
