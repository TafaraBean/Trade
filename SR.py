import pandas as pd
from main import bot
import matplotlib.pyplot as plt



# Define the date range
start = pd.Timestamp("2024-06-01")
end = pd.Timestamp("2024-06-29")

# Pulling the data from MT5
data = bot.copy_chart_range(symbol=bot.symbol, timeframe=bot.timeframe, start=start, end=end)

# Convert MT5 data to pandas DataFrame and set the time as the index
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

# Sort the data by time to avoid issues with plotting
data.sort_index(inplace=True)

# Function to find support and resistance
def find_support_resistance(data, window= 5 , threshold=0.003):
    support = []
    resistance = []
    recent_support = set()
    recent_resistance = set()

    for i in range(window, len(data)-window):
        local_min = min(data['close'][i-window:i+window])
        local_max = max(data['close'][i-window:i+window])

        # If the current point is the local minimum or maximum
        if data['close'][i] == local_min:
            if all(abs(local_min - s) > threshold for s in recent_support):
                support.append((data.index[i], data['close'][i]))
                recent_support.add(local_min)

        if data['close'][i] == local_max:
            if all(abs(local_max - r) > threshold for r in recent_resistance):
                resistance.append((data.index[i], data['close'][i]))
                recent_resistance.add(local_max)
        
    print(support.ty)
    
    return support, resistance


# Function to plot support and resistance lines
def plot_support_resistance(symbol, df, window=10):
    # Calculate support and resistance
    support, resistance = find_support_resistance(df, window)

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['close'], label=f'{symbol} Close Price', color='blue')

    # Plot support levels
    if support:
        support_prices = [sup[1] for sup in support]
        support_times = [sup[0] for sup in support]
        plt.hlines(support_prices, xmin=min(support_times), xmax=max(support_times), colors='green', linestyles='dashed', label='Support')

    # Plot resistance levels
    if resistance:
        resistance_prices = [res[1] for res in resistance]
        resistance_times = [res[0] for res in resistance]
        plt.hlines(resistance_prices, xmin=min(resistance_times), xmax=max(resistance_times), colors='red', linestyles='dashed', label='Resistance')

    plt.title(f'{symbol} - Support and Resistance Levels')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Example usage with your trading bot data

plot_support_resistance(bot.symbol, data)
