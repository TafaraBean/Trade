from datetime import datetime
import  MetaTrader5 as mt5 
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
timeframe = mt5.TIMEFRAME_D1
start = datetime(2024,5,11)
end = datetime.now()

data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)


df = pd.DataFrame(data)


# Convert 'date' column to datetime type
df['time'] = pd.to_datetime(df['time'],unit='s')
df['EMA'] = ta.ema(df['close'], length=3)
df['RSI'] = ta.rsi(df['close'], length=14)
df['SMA'] = ta.sma(df['close'], length=3)
print(df)
# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Candlestick')])

# Add SMA line
fig.add_trace(go.Scatter(x=df['time'], y=df['SMA'], 
                         mode='lines', name='SMA'))

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
#fig.show()
