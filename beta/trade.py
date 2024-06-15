from datetime import datetime
import  MetaTrader5 as mt5 
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from strategy import *
import time


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
timeframe = mt5.TIMEFRAME_H4
start = datetime(2024,6,1)
end = datetime.now()

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)

#create dataframe
df = pd.DataFrame(data)


# Convert 'date' column to datetime type
df['time'] = pd.to_datetime(df['time'],unit='s')

print(df)
df = h1_gold_strategy(df)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()
# Initialize tp and sl columns with None or NaN
filtered_df['tp'] = None
filtered_df['sl'] = None

# Calculate tp and sl for is_buy2 == True
filtered_df.loc[filtered_df['is_buy2'] == True, 'tp'] = filtered_df['low'] + 9
filtered_df.loc[filtered_df['is_buy2'] == True, 'sl'] = filtered_df['low'] - 3

# Calculate tp and sl for is_sell2 == True
filtered_df.loc[filtered_df['is_sell2'] == True, 'tp'] = filtered_df['high'] - 9
filtered_df.loc[filtered_df['is_sell2'] == True, 'sl'] = filtered_df['high'] + 3



df.to_csv('output.csv', index=False)

#gets the last row of the table
latest_signal=df.iloc[-1]


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