from datetime import datetime
import  MetaTrader5 as mt5 
from beta_trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from beta_strategy import *
import time
import numpy as np
from analysis import *


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
timeframe = mt5.TIMEFRAME_H1
start = datetime(2024,5,20)
end = datetime.now()

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)



df = h1_gold_strategy(data)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()
first_row_time = filtered_df.iloc[0]['time']
last_row_time = filtered_df.iloc[-1]['time'] 
last_row_time  += pd.Timedelta(hours=5)

count = 0
successful_trades = 0
unsuccessful_trades = 0

ticks = bot.get_ticks(symbol=symbol,start=start,end=end)

for index, row in filtered_df.iterrows():
    # Access specific columns of the current row
    
    time_value = row['time']
    end_date = datetime.now()
    sl = row['sl']  # Stop loss for this trade
    tp = row['tp']  # Take profit for this trade
    
    # Filter ticks dataframe from time_value onwards
    relevant_ticks = bot.get_ticks(symbol=symbol,start=time_value,end=end_date)
    print 

    
    # Check if stop loss or take profit was reached first
    stop_loss_reached = relevant_ticks['bid'] <= sl
    take_profit_reached = relevant_ticks['bid'] >= tp



    if any(stop_loss_reached):
        stop_loss_index = np.argmax(relevant_ticks['bid']  <= sl)
        #print(f"sl index:{stop_loss_index} ... Trade {row["time"]}")
        

    if any(take_profit_reached):
        take_profit_index = np.argmax(relevant_ticks['bid']  >= tp)
        #print(f"tp index:{take_profit_index} ... Trade {row["time"]}")


    # Compare which was reached first
    if stop_loss_reached.any() and take_profit_reached.any():
        if stop_loss_index < take_profit_index:
            unsuccessful_trades+=1
        else:
            successful_trades+=1
            bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=0.01, distance=900)
    elif stop_loss_reached.any():
        unsuccessful_trades+=1
    elif take_profit_reached.any():
        successful_trades+=1
    else:
        print(f"Neither stop loss nor take profit was reached.")

print(f"Total successful trades: {successful_trades}")
print(f"Total unsuccessful trades: {unsuccessful_trades}")

ticks = bot.get_ticks(symbol=symbol,start=start,end=end)

filtered_df.to_csv('beta/output.csv', index=False)
ticks.to_csv('beta/ticks.csv', index=False)

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
#fig.show()