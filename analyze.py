from datetime import datetime
import  MetaTrader5 as mt5 
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from strategy import *


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
timeframe = mt5.TIMEFRAME_H1
start = datetime(2024,5,1)
end = datetime.now()

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)

#create dataframe
df = pd.DataFrame(data)


# Convert 'date' column to datetime type
df['time'] = pd.to_datetime(df['time'],unit='s')


df = apply_strategy(df)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)]


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

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)]
filtered_df.to_csv('output.csv', index=False)




total_profit = 0
total_loss = 0

# Loop through each row using iterrows()
for index, row in df.iterrows():
    
    if row['is_buy2'] and row['take_profit_hit']:
        # If it is a buy
        total_profit = total_profit+ bot.cal_profit(symbol="XAUUSD", order_type=mt5.ORDER_TYPE_BUY, lot=0.01, distance=700)
    elif row['is_sell2'] and row['take_profit_hit']:
        # If it is a sell (assuming you want to do something similar for sells)
        total_profit = total_profit+ bot.cal_profit(symbol="XAUUSD", order_type=mt5.ORDER_TYPE_SELL, lot=0.01, distance=700)

for index, row in df.iterrows():
    
    if row['is_buy2'] and row['stop_loss_hit']:
        # If it is a buy
        total_loss = total_loss+ bot.cal_profit(symbol="XAUUSD", order_type=mt5.ORDER_TYPE_BUY, lot=0.01, distance=-300)
    elif row['is_sell2'] and row['stop_loss_hit']:
        # If it is a sell (assuming you want to do something similar for sells)
        total_loss = total_loss+ bot.cal_profit(symbol="XAUUSD", order_type=mt5.ORDER_TYPE_SELL, lot=0.01, distance=-300)

print(f"total profit is: {total_profit} {bot.account.currency}")
print(f"total loss is: {total_loss} {bot.account.currency}")


profit_factor = total_profit / abs(total_loss)

# Total number of trades
total_trades = len(filtered_df)

# Number of profitable trades (where take_profit_hit is not None)
profitable_trades = df['take_profit_hit'].notna().sum()

# Percentage profitability
percentage_profitability = (profitable_trades / total_trades) * 100

print(f"precentage profitability: {percentage_profitability}%")

print("Profit Factor:", profit_factor)
#print(filtered_df)
#print(calc_prof())

#gets the last row of the table
latest_signal=df.iloc[-1]

#checks if a buy/sell condition was met
if latest_signal["is_buy2"]:
  bot.open_buy_order(symbol=symbol,volume=0.01,tp=latest_signal['low']+9,sl=latest_signal['low']-3)
elif latest_signal["is_sell2"]:
  bot.open_sell_order(symbol=symbol,volume=0.01,tp=latest_signal['high']-9,sl=latest_signal['high']+3)



# Create candlestick chart
# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Candlestick')])

# Add buy signals (up arrows)
fig.add_trace(go.Scatter(
    x=df[df['is_buy2'] == True]['time'],
    y=df[df['is_buy2'] == True]['low'] * 0.999,  # Place the arrow slightly below the low price of the candle
    mode='markers',
    marker=dict(symbol='arrow-up', color='green', size=10),
    name='Buy Signal'
))



# Add sell signals (down arrows)
fig.add_trace(go.Scatter(
    x=df[df['is_sell2'] == True]['time'],
    y=df[df['is_sell2'] == True]['high'] * 1.001,  # Place the arrow slightly above the high price of the candle
    mode='markers',
    marker=dict(symbol='arrow-down', color='red', size=10),
    name='Sell Signal'
))

# Add LMSA Upper Band line
fig.add_trace(go.Scatter(x=df['time'], y=df['lsma_upper_band'], 
                         mode='lines', name='LMSA Upper Band'))

# Add LMSA Lower Band line
fig.add_trace(go.Scatter(x=df['time'], y=df['lsma_lower_band'], 
                         mode='lines', name='LMSA Lower Band'))

# Add LMSA  Band line
fig.add_trace(go.Scatter(x=df['time'], y=df['lsma'], 
                         mode='lines', name='LMSA'))

# Update layout
fig.update_layout(title='XAUUSD',
                  xaxis_title='Date',
                  yaxis_title='Price')

fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
    ]
)

# Show the plot
fig.show()