# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance
# pip install statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


PRICE_VALUE = "Close"
# Load and preprocess data
df = pd.read_csv('CBA.AX_2020-01-01_2023-07-31.csv', index_col='Date', parse_dates=True)
df = df[PRICE_VALUE]

# Train-test split
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# ARIMA model
arima_model = ARIMA(train_data, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_predictions = arima_model_fit.forecast(steps=len(test_data))

#ARIMA predictions alignment
arima_predictions_aligned = np.array(arima_predictions[:len(test_data)])
print("ARIMA Predictions:", arima_predictions_aligned)

# SARIMA model
sarima_model = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()
sarima_predictions = sarima_model_fit.forecast(steps=len(test_data))

#SARIMA predictions alignment
sarima_predictions_aligned = np.array(sarima_predictions[:len(test_data)])
print("SARIMA Predictions:", sarima_predictions_aligned)

# LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(np.array(train_data).reshape(-1, 1))

PREDICTION_DAYS = 60

X_train = []
y_train = []

for i in range(PREDICTION_DAYS, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-PREDICTION_DAYS:i, 0])
    y_train.append(scaled_train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(units=25))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

#LSTM model
scaled_test_data = scaler.transform(np.array(test_data).reshape(-1, 1))

X_test = []
y_test = df[train_size-PREDICTION_DAYS:train_size]

for i in range(PREDICTION_DAYS, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-PREDICTION_DAYS:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Align LSTM predictions with ARIMA
lstm_predictions_aligned = lstm_predictions[:len(test_data)]
print("LSTM Predictions:", lstm_predictions_aligned.flatten())

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(np.arange(len(train_data)).reshape(-1, 1), train_data)
rf_predictions = rf_model.predict(np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1))

# Random forest predictions alignment
rf_predictions_aligned = rf_predictions[:len(test_data)]

# Ensemble (average of ARIMA, SARIMA, and LSTM predictions)
min_len = min(len(arima_predictions_aligned), len(sarima_predictions_aligned), len(lstm_predictions_aligned))
ensemble_predictions = (arima_predictions_aligned[:min_len] + sarima_predictions_aligned[:min_len] + lstm_predictions_aligned[:min_len].flatten()) / 3
print("Ensemble Predictions:", ensemble_predictions)



# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[:min_len], test_data.values[:min_len], color="blue", label="Actual Prices")
plt.plot(test_data.index[:min_len], arima_predictions_aligned[:min_len], color="green", label="ARIMA Predictions", linewidth=2)
plt.plot(test_data.index[:min_len], sarima_predictions_aligned[:min_len], color="purple", label="SARIMA Predictions")
plt.plot(test_data.index[:len(lstm_predictions_aligned)], lstm_predictions_aligned, color="orange", label="LSTM Predictions")
plt.plot(test_data.index[:len(rf_predictions_aligned)], rf_predictions_aligned, color="brown", label="Random Forest Predictions")
plt.plot(test_data.index[:min_len], ensemble_predictions, color="red", label="Ensemble Predictions")
plt.title(f"Stock Price Prediction Ensemble: ARIMA + SARIMA + LSTM + Random Forest for CBA.AX")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??