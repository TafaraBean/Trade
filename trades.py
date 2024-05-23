import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Fetching historical data for EUR/USD
dataF = yf.download("EURUSD=X", start="2024-04-07", end="2024-05-05", interval='1h')

# Calculate EMAs
dataF['EMA_short'] = dataF['Close'].ewm(span=12, adjust=False).mean()
dataF['EMA_long'] = dataF['Close'].ewm(span=26, adjust=False).mean()

# Identify local minima and maxima
dataF['min'] = dataF.iloc[argrelextrema(dataF['Close'].values, np.less_equal, order=5)[0]]['Close']
dataF['max'] = dataF.iloc[argrelextrema(dataF['Close'].values, np.greater_equal, order=5)[0]]['Close']

# Function to check if a level is significant
def is_significant_level(levels, level, price_range):
    return np.any(np.abs(levels - level) < price_range)

support_levels = []
resistance_levels = []

price_range = np.mean(dataF['Close']) * 0.005  # 0.5% threshold for significance

for index, row in dataF.iterrows():
    if not np.isnan(row['min']):
        if not is_significant_level(np.array(support_levels), row['min'], price_range):
            support_levels.append(row['min'])
    if not np.isnan(row['max']):
        if not is_significant_level(np.array(resistance_levels), row['max'], price_range):
            resistance_levels.append(row['max'])

# Ensure there's enough data
if len(dataF) < 2:
    raise ValueError("Not enough data to generate signals.")

# Preallocate signal array with zeros
signals = [0] * len(dataF)

# Generate signals based on patterns and EMA conditions
for i in range(1, len(dataF)):
    current_open, current_close = dataF.Open.iloc[i], dataF.Close.iloc[i]
    previous_open, previous_close = dataF.Open.iloc[i-1], dataF.Close.iloc[i-1]

    if (current_open > current_close and 
        previous_open < previous_close and 
        current_close < previous_open and 
        current_open >= previous_close and 
        dataF['EMA_short'].iloc[i] > dataF['EMA_long'].iloc[i]):
        signals[i] = 1  # Bearish pattern

    elif (current_open < current_close and 
          previous_open > previous_close and 
          current_close > previous_open and 
          current_open <= previous_close and 
          dataF['EMA_short'].iloc[i] < dataF['EMA_long'].iloc[i]):
        signals[i] = 2  # Bullish pattern

# Assign signals to DataFrame
dataF['signal'] = signals

# Plotting the closing price with EMAs, signals, and support/resistance levels
plt.figure(figsize=(14, 7))
plt.plot(dataF.index, dataF['Close'], label='Close Price')
plt.plot(dataF.index, dataF['EMA_short'], label='12-Period EMA', alpha=0.7)
plt.plot(dataF.index, dataF['EMA_long'], label='26-Period EMA', alpha=0.7)

# Highlighting the signals
bullish_signals = dataF[dataF['signal'] == 2]
bearish_signals = dataF[dataF['signal'] == 1]

plt.scatter(bullish_signals.index, bullish_signals['Close'], marker='^', color='g', label='Bullish Signal', s=100)
plt.scatter(bearish_signals.index, bearish_signals['Close'], marker='v', color='r', label='Bearish Signal', s=100)

# Plotting support and resistance levels
for level in support_levels:
    plt.axhline(y=level, color='b', linestyle='--', label='Support Level' if level == support_levels[0] else "")
for level in resistance_levels:
    plt.axhline(y=level, color='r', linestyle='--', label='Resistance Level' if level == resistance_levels[0] else "")

# Adding labels and legend
plt.title('EUR/USD Close Price with EMA, Signals, and Support/Resistance Levels')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
