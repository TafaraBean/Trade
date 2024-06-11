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
timeframe = mt5.TIMEFRAME_M15
start = datetime(2024,6,11)
end = datetime.now()

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)

#create dataframe
df = pd.DataFrame(data)


# Convert 'date' column to datetime type
df['time'] = pd.to_datetime(df['time'],unit='s')


df = apply_strategy(df)
print(df)

#gets the last row of the table
latest_signal=df.iloc[-1]

#checks if a buy/sell condition was met
if latest_signal["is_buy2"]:
  bot.open_buy_order(symbol=symbol,volume=0.01)
elif latest_signal["is_sell2"]:
  bot.open_sell_order(symbol=symbol,volume=0.01)





# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Candlestick')])


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
