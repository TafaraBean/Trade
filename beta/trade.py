from datetime import datetime, timedelta
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

timeframe_to_interval = {
            mt5.TIMEFRAME_M1: "min",
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_M10: "10min",
            mt5.TIMEFRAME_M15: "15min",
            mt5.TIMEFRAME_M30: "30min",
            mt5.TIMEFRAME_H1: "H",
            mt5.TIMEFRAME_H4: "4H",
            mt5.TIMEFRAME_D1: "D",
        }


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
timeframe = mt5.TIMEFRAME_H1
start = pd.to_datetime(datetime(2024,5,1))
conversion = timeframe_to_interval.get(timeframe, 3600)
end = (pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)



df = h1_gold_strategy(data)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()
first_row_time = filtered_df.iloc[0]['time']
last_row_time = filtered_df.iloc[-1]['time'] 
last_row_time  += pd.Timedelta(hours=5)

unexecuted_trades = 0
successful_trades = 0
unsuccessful_trades = 0
gross_profit = 0
loss = 0
total_trades = 0
num_winning_trades = 0
executed_trades = [] 


for index, row in filtered_df.iterrows():
    # Access specific columns of the current row
    
    time_value = row['time']
    end_date =  datetime.now() + timedelta(hours=1)
    sl = row['sl']  # Stop loss for this trade
    tp = row['tp']  # Take profit for this trade
    
    # Filter ticks dataframe from time_value onwards
    relevant_ticks = bot.get_ticks(symbol=symbol,start=time_value,end=end_date) 

    
    # Check if stop loss or take profit was reached first
    stop_loss_reached = relevant_ticks['bid'] <= sl
    take_profit_reached = relevant_ticks['bid'] >= tp


    if any(stop_loss_reached):
        stop_loss_index = np.argmax(stop_loss_reached)
    else:
        stop_loss_index = -1

    if any(take_profit_reached):
        take_profit_index = np.argmax(stop_loss_reached)
    else:
        take_profit_index = -1

    if take_profit_index == 0 or stop_loss_index == 0:
        unexecuted_trades +=1
        continue
    # Compare which was reached first
    successful = False

    total_trades+=1
    if stop_loss_reached.any() and take_profit_reached.any():
        if stop_loss_index < take_profit_index:
            unsuccessful_trades+=1
            if row["is_buy2"]:
                gross_profit += bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=0.01, open_price=row["close"], close_price=row["sl"])
            else: 
                gross_profit += bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=0.01, open_price=row["close"], close_price=row["sl"])
        else:
            successful = True
            successful_trades+=1
            num_winning_trades +=1
            if row["is_buy2"]:
                gross_profit += bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=0.01, open_price=row["close"], close_price=row["tp"])
            else: 
                gross_profit += bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=0.01, open_price=row["close"], close_price=row["tp"])
    elif stop_loss_reached.any():
        unsuccessful_trades+=1
        if row["is_buy2"]:
            loss += bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=0.01, open_price=row["close"], close_price=row["sl"])
        else: 
            loss += bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=0.01, open_price=row["close"], close_price=row["sl"])
    elif take_profit_reached.any():
        successful = True
        successful_trades+=1
        num_winning_trades +=1
        if row["is_buy2"]:
            gross_profit += bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=0.01, open_price=row["close"], close_price=row["tp"])
        else: 
            gross_profit += bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=0.01, open_price=row["close"], close_price=row["tp"])
    else:
        print(f"Neither stop loss nor take profit was reached.")

    row['successful'] = successful
    executed_trades.append(row)
if loss != 0:
    profit_factor = gross_profit / abs(loss)
else:
    profit_factor = float('inf')  # Handle case where there are no losing trades

if total_trades > 0:
    percentage_profitability = (num_winning_trades / total_trades) * 100
else:
    percentage_profitability = 0  # Handle case where there are no trades


executed_trades_df = pd.DataFrame(executed_trades)


filtered_df.to_csv('beta/filtered_df.csv', index=False)
executed_trades_df.to_csv('beta/executed_trades_df.csv', index=False)


print(f"analysis from {start} to {end}")
print(f"Total unexecuted trades: {unexecuted_trades}")
print(f"Total successful trades: {successful_trades}")
print(f"Total unsuccessful trades: {unsuccessful_trades}")
print(f"gross profit: {gross_profit} {bot.account.currency}" )
print(f"loss: {loss} {bot.account.currency}")
print(f"net proft: {gross_profit+loss}")
print(f"proft factor: {profit_factor}")

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['time'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Candlestick')])
"""
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
"""

for index, row in executed_trades_df.iterrows():
        # Add green rectangle for take profit
        fig.add_shape(
            type="rect",
            x0=row['time'], x1=row['time'] + pd.Timedelta(hours=4),
            y0=row['close'], y1=row['tp'],
            line=dict(color="green", width=2),
            fillcolor="green",
            opacity=0.3
        )
        # Add red rectangle for stop loss
        fig.add_shape(
            type="rect",
            x0=row['time'], x1=row['time'] + pd.Timedelta(hours=4),
            y0=row['sl'], y1=row['close'],
            line=dict(color="red", width=2),
            fillcolor="red",
            opacity=0.3
        )

# Add buy signals (up arrows)
fig.add_trace(go.Scatter(
    x=executed_trades_df[executed_trades_df['is_buy2'] == True]['time'],
    y=executed_trades_df[executed_trades_df['is_buy2'] == True]['low'] * 0.999,  # Place the arrow slightly below the low price of the candle
    mode='markers',
    marker=dict(symbol='arrow-up', color='green', size=10),
    name='Buy Signal'
))


# Add sell signals (down arrows)
fig.add_trace(go.Scatter(
    x=executed_trades_df[executed_trades_df['is_sell2'] == True]['time'],
    y=executed_trades_df[executed_trades_df['is_sell2'] == True]['high'] * 1.001,  # Place the arrow slightly above the high price of the candle
    mode='markers',
    marker=dict(symbol='arrow-down', color='red', size=10),
    name='Sell Signal'
))


# Update layout
fig.update_layout(title='XAUUSD',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  #template="plotly_dark"
                  )

fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
    ]
)

fig.update_yaxes(type="log")


# Show the plot
fig.show()

del relevant_ticks
del filtered_df
del df