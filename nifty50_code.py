import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

# Load the model
model = load_model('LSTM_NIFTY_10(169.54).h5')

# Function to create dataset
def create_dataset(dataset, look_back=1):
    X = []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
    return np.array(X)

# Function to make predictions
def predict_future(data, look_back, num_days):
    data_scaled = scaler.transform(data.reshape(-1, 1))
    X_input = create_dataset(data_scaled, look_back)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    
    predictions = []
    for _ in range(num_days):
        pred = model.predict(X_input[-1].reshape(1, look_back, 1))
        predictions.append(pred[0, 0])
        X_input = np.append(X_input, pred, axis=0)
        X_input = X_input[1:]  # Shift the input window

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Set up Streamlit UI
st.title("NIFTY50 Price Prediction")

# User input for date range
start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('today'))

if st.button("Fetch Data"):
    # Fetch historical data from Yahoo Finance
    ticker = '^NSEI'  # NIFTY50 ticker symbol
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    st.write("Historical Data:")
    st.write(df)

    # Prepare the data
    data = df['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    look_back = 10  # Use the same look_back used during training
    num_days = st.number_input("Enter the number of days to predict:", min_value=1, max_value=30, value=5)

    if st.button("Predict"):
        predictions = predict_future(data_scaled, look_back, num_days)
        st.write("Predicted Prices for the next days:")
        st.write(predictions)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(data), len(data) + num_days), predictions, label='Predicted Future Prices', marker='o')
        plt.title('Future Price Predictions')
        plt.xlabel('Days')
        plt.ylabel('NIFTY50 Price')
        plt.legend()
        st.pyplot(plt)
