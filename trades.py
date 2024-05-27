import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import sched
import time

# Alpaca API credentials
API_KEY = 'AK1BT547IBID17JK3BV4'
API_SECRET = '0osEiQ9wGKjuqjdzQ4KQG4bb0fSyNXevi6V7gNC2'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Scheduler setup
scheduler = sched.scheduler(time.time, time.sleep)

# Function to fetch data and process signals
def fetch_and_process_data():
    # Fetching historical data for the last 18 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=18)
    start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fetch data
    bars = api.get_bars('GLD', tradeapi.rest.TimeFrame.Minute, start=start_str, end=end_str, adjustment='raw',feed='iex').df
    bars = bars.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate EMAs
    bars['EMA_short'] = bars['close'].ewm(span=12, adjust=False).mean()
    bars['EMA_long'] = bars['close'].ewm(span=26, adjust=False).mean()

    # Calculate Bollinger Bands
    bars['SMA'] = bars['close'].rolling(window=20).mean()
    bars['stddev'] = bars['close'].rolling(window=20).std()
    bars['Upper_Band'] = bars['SMA'] + (bars['stddev'] * 2)
    bars['Lower_Band'] = bars['SMA'] - (bars['stddev'] * 2)

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

    bars['RSI'] = calculate_RSI(bars['close'], period=14)

    # Identify local minima and maxima
    bars['min'] = bars.iloc[argrelextrema(bars['close'].values, np.less_equal, order=5)[0]]['close']
    bars['max'] = bars.iloc[argrelextrema(bars['close'].values, np.greater_equal, order=5)[0]]['close']

    # Function to check if a level is significant
    def is_significant_level(levels, level, price_range):
        return np.any(np.abs(levels - level) < price_range)

    support_levels = []
    resistance_levels = []

    price_range = np.mean(bars['close']) * 0.01  # 1% threshold for significance

    for index, row in bars.iterrows():
        if not np.isnan(row['min']):
            if not is_significant_level(np.array(support_levels), row['min'], price_range):
                support_levels.append(row['min'])
        if not np.isnan(row['max']):
            if not is_significant_level(np.array(resistance_levels), row['max'], price_range):
                resistance_levels.append(row['max'])

    # Only keep the most recent levels
    recent_support_levels = support_levels[-5:]  # Last 5 support levels
    recent_resistance_levels = resistance_levels[-5:]  # Last 5 resistance levels

    # Function to determine the area (support, resistance, or neither)
    def determine_area(close, support_levels, resistance_levels, threshold):
        if any(abs(close - level) < threshold for level in support_levels):
            return 'Support'
        elif any(abs(close - level) < threshold for level in resistance_levels):
            return 'Resistance'
        else:
            return 'Neither'

    # Determine area for each row
    threshold = np.mean(bars['close']) * 0.005  # 0.5% threshold
    bars['Area'] = bars['close'].apply(lambda x: determine_area(x, recent_support_levels, recent_resistance_levels, threshold))

    # Ensure there's enough data
    if len(bars) < 2:
        raise ValueError("Not enough data to generate signals.")

    # Preallocate signal array with zeros
    signals = [0] * len(bars)

    # Generate signals based on candle patterns, EMA conditions, and support/resistance levels
    for i in range(1, len(bars)):
        current_open, current_close = bars.open.iloc[i], bars.close.iloc[i]
        previous_open, previous_close = bars.open.iloc[i-1], bars.close.iloc[i-1]
        previous_volume, current_volume = bars.volume.iloc[i-1], bars.volume.iloc[i]
        current_area = bars['Area'].iloc[i]
        ema_short, ema_long = bars['EMA_short'].iloc[i], bars['EMA_long'].iloc[i]
        rsi = bars['RSI'].iloc[i]

        # Debug print statements
        print(f"Index: {i}, Current Close: {current_close}, Current Volume: {current_volume}, Area: {current_area}")

        # Additional conditions for pullbacks and retests
        is_pullback = False
        is_retest = False

        # Check for pullback: temporary reversal in trend
        if ema_short > ema_long:  # Uptrend
            if current_close < previous_close and previous_close > bars['close'].iloc[i-2]:
                is_pullback = True
        elif ema_short < ema_long:  # Downtrend
            if current_close > previous_close and previous_close < bars['close'].iloc[i-2]:
                is_pullback = True

        # Check for retest: revisit of support or resistance
        if current_area == 'Support' or current_area == 'Resistance':
            if current_area == 'Support' and previous_close < current_close:
                is_retest = True
            elif current_area == 'Resistance' and previous_close > current_close:
                is_retest = True

        if (current_area == 'Support' or current_area == 'Resistance'):
            if current_volume > previous_volume and current_open < current_close and previous_open > previous_close and ema_short > ema_long and current_close > previous_open:
                if signals[i-1] != 2:  # Avoid consecutive bullish signals
                    signals[i] = 2  # Bullish signal
                    print(f"Bullish Signal at index {i}")

            elif current_volume > previous_volume and current_open > current_close and previous_open < previous_close and ema_short < ema_long and current_close < previous_open:
                if signals[i-1] != 1:  # Avoid consecutive bearish signals
                    signals[i] = 1  # Bearish signal
                    print(f"Bearish Signal at index {i}")

    # Assign signals to DataFrame
    bars['signal'] = signals

    # Calculate accuracy
    correct_signals = 0
    total_signals = 0

    for i in range(1, len(bars)):
        if signals[i] != 0:
            total_signals += 1
            j = i + 1
            while j < len(bars) and signals[j] == 0:
                j += 1
            if j < len(bars):
                if signals[i] == 2 and bars['close'].iloc[i] < bars['close'].iloc[j]:
                    correct_signals += 1
                elif signals[i] == 1 and bars['close'].iloc[i] > bars['close'].iloc[j]:
                    correct_signals += 1

    accuracy = correct_signals / total_signals * 100 if total_signals > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")

    # Plotting the closing price with EMAs, signals, Bollinger Bands, and support/resistance levels
    plt.figure(figsize=(14, 7))
    plt.plot(bars.index, bars['close'], label='Close Price')
    plt.plot(bars.index, bars['EMA_short'], label='12-Period EMA', alpha=0.7)
    plt.plot(bars.index, bars['EMA_long'], label='26-Period EMA', alpha=0.7)
    plt.plot(bars.index, bars['SMA'], label='20-Period SMA', alpha=0.7)
    plt.plot(bars.index, bars['Upper_Band'], label='Upper Bollinger Band', linestyle='--', alpha=0.7)
    plt.plot(bars.index, bars['Lower_Band'], label='Lower Bollinger Band', linestyle='--', alpha=0.7)

    # Highlighting the signals
    bullish_signals = bars[bars['signal'] == 2]
    bearish_signals = bars[bars['signal'] == 1]

    plt.scatter(bullish_signals.index, bullish_signals['close'], marker='^', color='g', label='Bullish Signal', s=100)
    plt.scatter(bearish_signals.index, bearish_signals['close'], marker='v', color='r', label='Bearish Signal', s=100)

    # Plotting the most recent support and resistance levels
    for level in recent_support_levels:
        plt.axhline(y=level, color='b', linestyle='--', label='Support Level' if level == recent_support_levels[0] else "")
    for level in recent_resistance_levels:
        plt.axhline(y=level, color='r', linestyle='--', label='Resistance Level' if level == recent_resistance_levels[0] else "")

    # Adding labels and legend
    plt.title('SPY Close Price with EMA, Bollinger Bands, Signals, and Support/Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save to Excel for further analysis
    bars.index = bars.index.tz_localize(None)
    bars[['RSI', 'EMA_short', 'EMA_long', 'Upper_Band', 'Lower_Band', 'Area', 'signal']].to_excel('trading_signals.xlsx', index=True, float_format='%.5f')

# Schedule the fetch_and_process_data function to run every 15 minutes
scheduler.enter(0, 1, fetch_and_process_data)
scheduler.enter(900, 1, fetch_and_process_data)
scheduler.run()
