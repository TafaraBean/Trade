import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Fetching historical data for EUR/USD
dataF = yf.download("GC=F", start="2024-05-07", end="2024-05-25", interval='30m')

# Calculate EMAs
dataF['EMA_short'] = dataF['Close'].ewm(span=12, adjust=False).mean()
dataF['EMA_long'] = dataF['Close'].ewm(span=26, adjust=False).mean()

# Calculate Bollinger Bands
dataF['SMA'] = dataF['Close'].rolling(window=20).mean()
dataF['stddev'] = dataF['Close'].rolling(window=20).std()
dataF['Upper_Band'] = dataF['SMA'] + (dataF['stddev'] * 2)
dataF['Lower_Band'] = dataF['SMA'] - (dataF['stddev'] * 2)

# Calculate RSI
def calculate_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

dataF['RSI'] = calculate_RSI(dataF['Close'], period=14)

# Identify local minima and maxima
dataF['min'] = dataF.iloc[argrelextrema(dataF['Close'].values, np.less_equal, order=5)[0]]['Close']
dataF['max'] = dataF.iloc[argrelextrema(dataF['Close'].values, np.greater_equal, order=5)[0]]['Close']

# Function to check if a level is significant
def is_significant_level(levels, level, price_range):
    return np.any(np.abs(levels - level) < price_range)

support_levels = []
resistance_levels = []

price_range = np.mean(dataF['Close']) * 0.01  # 1% threshold for significance

for index, row in dataF.iterrows():
    if not np.isnan(row['min']):
        if not is_significant_level(np.array(support_levels), row['min'], price_range):
            support_levels.append(row['min'])
    if not np.isnan(row['max']):
        if not is_significant_level(np.array(resistance_levels), row['max'], price_range):
            resistance_levels.append(row['max'])

# Only keep the most recent levels
recent_support_levels = support_levels[-5:]  # Last 3 support levels
recent_resistance_levels = resistance_levels[-5:]  # Last 3 resistance levels

# Function to determine the area (support, resistance, or neither)
def determine_area(close, support_levels, resistance_levels, threshold):
    if any(abs(close - level) < threshold for level in support_levels):
        return 'Support'
    elif any(abs(close - level) < threshold for level in resistance_levels):
        return 'Resistance'
    else:
        return 'Neither'

# Determine area for each row
threshold = np.mean(dataF['Close']) * 0.005  # 0.5% threshold
dataF['Area'] = dataF['Close'].apply(lambda x: determine_area(x, recent_support_levels, recent_resistance_levels, threshold))

# Ensure there's enough data
if len(dataF) < 2:
    raise ValueError("Not enough data to generate signals.")

# Preallocate signal array with zeros
signals = [0] * len(dataF)

# Generate signals based on candle patterns, EMA conditions, and support/resistance levels
for i in range(1, len(dataF)):
    current_open, current_close = dataF.Open.iloc[i], dataF.Close.iloc[i]
    previous_open, previous_close = dataF.Open.iloc[i-1], dataF.Close.iloc[i-1]
    previous_volume, current_volume = dataF.Volume.iloc[i-1], dataF.Volume.iloc[i]
    current_area = dataF['Area'].iloc[i]
    ema_short, ema_long = dataF['EMA_short'].iloc[i], dataF['EMA_long'].iloc[i]
    rsi=dataF['RSI'].iloc[i]

    # Debug print statements
    print(f"Index: {i}, Current Close: {current_close}, Current Volume: {current_volume}, Area: {current_area}")

    if current_area == 'Support' or current_area == 'Resistance':
        if current_volume > previous_volume and current_open < current_close and previous_open>previous_close and ema_short > ema_long:
            if signals[i-1] != 2:  # Avoid consecutive bullish signals
                signals[i] = 2  # Bullish signal
                print(f"Bullish Signal at index {i}")

        elif current_volume > previous_volume and current_open > current_close and previous_open<previous_close and ema_short < ema_long:
            if signals[i-1] != 1:  # Avoid consecutive bearish signals
                signals[i] = 1  # Bearish signal
                print(f"Bearish Signal at index {i}")

# Assign signals to DataFrame
dataF['signal'] = signals

# Calculate accuracy
correct_signals = 0
total_signals = 0

for i in range(1, len(dataF)):
    if signals[i] != 0:
        total_signals += 1
        j = i + 1
        while j < len(dataF) and signals[j] == 0:
            j += 1
        if j < len(dataF):
            if signals[i] == 2 and dataF['Close'].iloc[i] < dataF['Close'].iloc[j]:
                correct_signals += 1
            elif signals[i] == 1 and dataF['Close'].iloc[i] > dataF['Close'].iloc[j]:
                correct_signals += 1

accuracy = correct_signals / total_signals * 100 if total_signals > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")

# Plotting the closing price with EMAs, signals, Bollinger Bands, and support/resistance levels
plt.figure(figsize=(14, 7))
plt.plot(dataF.index, dataF['Close'], label='Close Price')
plt.plot(dataF.index, dataF['EMA_short'], label='12-Period EMA', alpha=0.7)
plt.plot(dataF.index, dataF['EMA_long'], label='26-Period EMA', alpha=0.7)
plt.plot(dataF.index, dataF['SMA'], label='20-Period SMA', alpha=0.7)
plt.plot(dataF.index, dataF['Upper_Band'], label='Upper Bollinger Band', linestyle='--', alpha=0.7)
plt.plot(dataF.index, dataF['Lower_Band'], label='Lower Bollinger Band', linestyle='--', alpha=0.7)

# Highlighting the signals
bullish_signals = dataF[dataF['signal'] == 2]
bearish_signals = dataF[dataF['signal'] == 1]

plt.scatter(bullish_signals.index, bullish_signals['Close'], marker='^', color='g', label='Bullish Signal', s=100)
plt.scatter(bearish_signals.index, bearish_signals['Close'], marker='v', color='r', label='Bearish Signal', s=100)

# Plotting the most recent support and resistance levels
for level in recent_support_levels:
    plt.axhline(y=level, color='b', linestyle='--', label='Support Level' if level == recent_support_levels[0] else "")
for level in recent_resistance_levels:
    plt.axhline(y=level, color='r', linestyle='--', label='Resistance Level' if level == recent_resistance_levels[0] else "")

# Adding labels and legend
plt.title('EUR/USD Close Price with EMA, Bollinger Bands, Signals, and Support/Resistance Levels')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

dataF.index = dataF.index.tz_localize(None)
# Save to Excel for further analysis
dataF[['RSI', 'EMA_short', 'EMA_long', 'Upper_Band', 'Lower_Band', 'Area', 'signal']].to_excel('trading_signals.xlsx', index=True, float_format='%.5f')
