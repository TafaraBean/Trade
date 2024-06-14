import pandas as pd

# Define the dataframe
df = pd.DataFrame({
    'time': ['2024-05-07 22:00:00', '2024-05-07 23:00:00', '2024-05-08 01:00:00', 
             '2024-05-08 02:00:00', '2024-05-08 03:00:00', '2024-05-17 19:00:00', 
             '2024-05-17 20:00:00', '2024-05-17 21:00:00', '2024-05-17 22:00:00', 
             '2024-05-17 23:00:00'],
    'open': [2314.48, 2314.57, 2313.43, 2314.42, 2315.17, 2407.73, 2407.77, 2414.23, 2417.28, 2416.54],
    'high': [2315.49, 2315.37, 2316.74, 2316.22, 2316.26, 2414.11, 2414.27, 2419.14, 2417.76, 2422.57],
    'low': [2311.75, 2312.48, 2313.18, 2314.12, 2313.71, 2407.63, 2405.66, 2413.18, 2411.41, 2413.46],
    'close': [2314.56, 2313.57, 2314.43, 2315.06, 2314.15, 2407.75, 2414.13, 2417.33, 2416.30, 2415.11],
    'tick_volume': [4218, 1539, 1602, 1467, 4551, 10959, 10377, 8949, 7104, 3387],
    'spread': [27, 30, 27, 27, 27, 27, 27, 27, 27, 30],
    'lsma': [None]*10,
    'macd_line': [None]*10,
    'lsma_stddev': [None]*10,
    'lsma_upper_band': [None]*10,
    'lsma_lower_band': [None]*10,
    'is_buy2': [False, False, False, True, False, False, False, False, False, False],
    'is_sell2': [True, False, False, False, False, False, True, False, False, False]
})

# Function to check if stop loss or take profit is reached
def check_stop_loss_take_profit(signal_index):
    trade_signal = df.iloc[signal_index]
    stop_loss = take_profit = None
    stop_loss_hit = take_profit_hit = None

    # Calculate stop loss and take profit
    if trade_signal['is_buy2']:
        stop_loss = trade_signal['low'] + 9
        take_profit = trade_signal['low'] - 3
    elif trade_signal['is_sell2']:
        stop_loss = trade_signal['high'] + 3
        take_profit = trade_signal['high'] - 9

    # Iterate over subsequent candles to check if stop loss or take profit is hit
    for i in range(signal_index + 1, len(df)):
        candle = df.iloc[i]
        if trade_signal['is_buy2']:
            if candle['low'] <= take_profit:
                take_profit_hit = df.index[i]
                break
            if candle['low'] <= stop_loss:
                stop_loss_hit = df.index[i]
                break
        elif trade_signal['is_sell2']:
            if candle['high'] >= stop_loss:
                stop_loss_hit = df.index[i]
                break
            if candle['high'] >= take_profit:
                take_profit_hit = df.index[i]
                break

    return stop_loss_hit, take_profit_hit

# Add new columns to store the results
df['stop_loss_hit'] = None
df['take_profit_hit'] = None

# Check each row for stop loss and take profit
for index in range(len(df)):
    if df.at[index, 'is_buy2'] or df.at[index, 'is_sell2']:
        stop_loss_hit, take_profit_hit = check_stop_loss_take_profit(index)
        df.at[index, 'stop_loss_hit'] = stop_loss_hit
        df.at[index, 'take_profit_hit'] = take_profit_hit

print(df)
