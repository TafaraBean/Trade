import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetching historical data for EUR/USD
dataF = yf.download("EURUSD=X", start="2024-04-7", end="2024-05-5", interval='15m')

# Ensure there's enough data
if len(dataF) < 2:
    raise ValueError("Not enough data to generate signals.")

# Preallocate signal array with zeros
signals = [0] * len(dataF)

# Generate signals based on patterns
for i in range(1, len(dataF)):
    open, close = dataF.Open.iloc[i], dataF.Close.iloc[i]
    previous_open, previous_close = dataF.Open.iloc[i-1], dataF.Close.iloc[i-1]

    if open > close and previous_open < previous_close and close < previous_open and open >= previous_close:
        signals[i] = 1  # Bearish pattern
    elif open < close and previous_open > previous_close and close > previous_open and open <= previous_close:
        signals[i] = 2  # Bullish pattern

# Add the signals to the DataFrame
dataF["signal"] = signals

# Plotting the closing price
plt.figure(figsize=(14, 7))
plt.plot(dataF.index, dataF['Close'], label='Close Price')

# Highlighting the signals
bullish_signals = dataF[dataF['signal'] == 2]
bearish_signals = dataF[dataF['signal'] == 1]

plt.scatter(bullish_signals.index, bullish_signals['Close'], marker='^', color='g', label='Bullish Signal', s=100)
plt.scatter(bearish_signals.index, bearish_signals['Close'], marker='v', color='r', label='Bearish Signal', s=100)

# Adding labels and legend
plt.title('EUR/USD Close Price with Trading Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
