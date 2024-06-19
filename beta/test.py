from datetime import datetime, timedelta
import  MetaTrader5 as mt5 
from beta_trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.graph_objects as go
from beta_strategy import *
import numpy as np
from analysis import *
from plotly.subplots import make_subplots

timeframe_to_interval = {
            mt5.TIMEFRAME_M1: "min",
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_M10: "10min",
            mt5.TIMEFRAME_M15: "15min",
            mt5.TIMEFRAME_M30: "30min",
            mt5.TIMEFRAME_H1: "h",
            mt5.TIMEFRAME_H4: "4h",
            mt5.TIMEFRAME_D1: "D",
        }


load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="XAUUSD"
account_balance = 300
lot_size = 0.01
timeframe = mt5.TIMEFRAME_M15
start = pd.to_datetime(datetime(2024,6,10))
conversion = timeframe_to_interval.get(timeframe, 3600)
end = (pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)



df = m15_gold_strategy(data.copy())

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()

total_trades = 0
unexecuted_trades = 0
successful_trades = 0
unsuccessful_trades = 0
gross_profit = 0
loss = 0
total_trades = 0
num_winning_trades = 0
executed_trades = [] 
biggest_loss = 0
biggest_win = 0

inital_balance = account_balance
max_drawdown = 1000


for index, row in filtered_df.iterrows():
    # Access specific columns of the current row
    start_time= (row['time'] + pd.Timedelta(minutes=2)).ceil(conversion) #add to minutes to be able to apply ceil function
    end_date =  pd.Timestamp.now() + pd.Timedelta(hours=1)    
    relevant_ticks = bot.get_ticks(symbol=symbol,start=start_time,end=end_date) # Filter ticks dataframe from time_value onwards
    

    #check if trade has invalid stopouts
    if check_invalid_stopouts(row):
        unexecuted_trades += 1
        print(f"trade invalid stopouts: {row['time']}")
        continue
    
    #check if candles ever closed 4 units above open
    print(f"Currently Working on Trade: {row['time']}")
    second_chart = bot.chart(symbol=symbol, timeframe=timeframe, start=start_time, end=end_date)
    
    if(row["is_buy2"]):
        add_trailing_stop = second_chart['close'] >= row["close"]+4
    else:
        add_trailing_stop = second_chart['close'] <= row["close"]-4


    # Check if stop loss or take profit was reached first
    if(row["is_buy2"]):
        stop_loss_reached = relevant_ticks['bid'] <= row["sl"]
        take_profit_reached = relevant_ticks['bid'] >= row["tp"]
    else:
        stop_loss_reached = relevant_ticks['bid'] >= row["sl"]
        take_profit_reached = relevant_ticks['bid'] <= row["tp"]

    #find the time at which it was4 units above entery

    add_trailing_stop_index = np.argmax(add_trailing_stop) if add_trailing_stop.any() else -1
    stop_loss_index = np.argmax(stop_loss_reached) if stop_loss_reached.any() else -1
    take_profit_index = np.argmax(take_profit_reached) if take_profit_reached.any() else -1

    time_to_trail = second_chart.loc[add_trailing_stop_index, "time"] if add_trailing_stop_index != -1 else None
    time_tp_hit = relevant_ticks.loc[take_profit_index, 'time'] if take_profit_index != -1 else None
    time_sl_hit = relevant_ticks.loc[stop_loss_index, 'time'] if stop_loss_index != -1 else None
    

    if(time_to_trail is not None and time_sl_hit is not None):
        if(time_to_trail < time_sl_hit):
            row['sl_updated'] = True
            row['time_updated'] = time_to_trail
            print(f"We trailled before stop loss hit")
            print(f"time to trail: {time_to_trail} \nTime to stopp: {time_sl_hit}\n")
        else:
            row['sl_updated'] = False
            row['time_updated'] = None
            print(f"stop loss hit before we could trail at")
            print(f"time to trail: {time_to_trail} \nTime to stopp: {time_sl_hit}\n")
    elif(time_to_trail is not None and time_sl_hit is None):
        row['sl_updated'] = True
        row['time_updated'] = time_to_trail
        print("sl never reached, only trailing sl added")
        print(f"time to trail: {time_to_trail} \nTime to stopp: {time_sl_hit}\n")

    else:
        row['sl_updated'] = False
        row['time_updated'] = None
        print("stop loss reached, and trailing never reahced")
        print(f"time to trail: {time_to_trail} \nTime to stopp: {time_sl_hit}\n")


    #update actual sl
    if  row['sl_updated']:
        if row['is_buy2']:
            row['sl'] = row['close'] + 3
        else:
            row['sl'] = row['close'] - 3

    
    #print(f"Trade: {row["time"]}\nsl: {stop_loss_index}\ntp: {take_profit_index}")
    if take_profit_index == 0 or stop_loss_index == 0:
        print(f"take profit or stop loss reached ar zero for trade {row['time']}")
        unexecuted_trades +=1
        continue
    

    # Compare which was reached first
    successful = False

    total_trades+=1
    
    if stop_loss_reached.any() and take_profit_reached.any():
        if(take_profit_index < stop_loss_index):
            #print("successful")
            #print(f"tp reached first at {relevant_ticks.loc[take_profit_index, 'time']}") 
            successful = True
            successful_trades+=1
            row['successful'] = successful
            row['position_close_time'] = relevant_ticks.loc[take_profit_index, 'time']
            if row["is_buy2"]:
                row['profit'] =  bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["tp"])
                gross_profit += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
            else: 
                row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["tp"])        
                gross_profit +=  row['profit'] 
                account_balance  += row['profit']
                row["account_balance"] = account_balance
        else:
            #print("unsuccessful")
            #print(f"sl reached first at {relevant_ticks.loc[stop_loss_index, 'time']}")
            unsuccessful_trades+=1
            row['successful'] = successful
            row['position_close_time'] = relevant_ticks.loc[stop_loss_index, 'time']
            if row["is_buy2"]:
                row['profit'] = bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["sl"])
                loss += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
            else: 
                row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["sl"])        
                loss += row["profit"]
                account_balance  += row['profit']
                row["account_balance"] = account_balance
    
    elif stop_loss_reached.any():
        #print("unsuccessful")
        #print(f"only sl reached at {relevant_ticks.loc[stop_loss_index, 'time']}")
        row['successful'] = successful
        row['position_close_time'] = relevant_ticks.loc[stop_loss_index, 'time']
        unsuccessful_trades+=1
        if row["is_buy2"]:
            row['profit'] = bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["sl"])
            loss += row['profit']
            account_balance  += row['profit']
            row["account_balance"] = account_balance
        else: 
            row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["sl"])        
            loss += row["profit"]
            account_balance  += row['profit']
            row["account_balance"] = account_balance
       
    elif take_profit_reached.any():
            #print("successful")
            #print(f"only tp reached at {relevant_ticks.loc[take_profit_index, 'time']}")
            row['successful'] = successful
            row['position_close_time'] = relevant_ticks.loc[take_profit_index, 'time']
            successful_trades+=1
            if row["is_buy2"]:
                row['profit'] =  bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["tp"])
                gross_profit += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
            else: 
                row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["tp"])        
                gross_profit +=  row['profit'] 
                account_balance  += row['profit']
                row["account_balance"] = account_balance
    else:
        row['position_close_time'] = row['time'] + pd.Timedelta(hours=3)
        print(f"Neither stop loss nor take profit was reached for trade. {row["time"]}")

    if account_balance < max_drawdown:
        max_drawdown = account_balance
    total_trades += 1
    row['successful'] = successful
    executed_trades.append(row)


if loss != 0:
    profit_factor = gross_profit / abs(loss)
else:
    profit_factor = float('inf')  # Handle case where there are no losing trades

if total_trades > 0:
    percentage_profitability = (successful_trades / (successful_trades+unsuccessful_trades)) * 100
else:
    percentage_profitability = 0  # Handle case where there are no trades


executed_trades_df = pd.DataFrame(executed_trades)

data.to_csv('beta/output.csv', index=False)
filtered_df.to_csv('beta/filtered_df.csv', index=False)
executed_trades_df.to_csv('beta/executed_trades_df.csv', index=False)
bot.get_ticks(symbol=symbol,start=start,end=end_date).to_csv("beta/ticks.csv", index=False)

print(f"\nanalysis from {start} to {end}\n")
print(f"\nPROFITABILITY\n")
print(f"Total unexecuted trades: {unexecuted_trades}")
print(f"Total successful trades: {successful_trades}") 
print(f"Total unsuccessful trades: {unsuccessful_trades}")
print(f"gross profit: {round(gross_profit, 2)} {bot.account.currency}")
print(f"loss: {round(loss, 2)} {bot.account.currency}")
print(f"percentage profitability: {percentage_profitability} %")

print(f"profit factor: {round(profit_factor, 2)}")
print(f"\nACCOUNT DETAILS\n")
print(f"lot size used: {lot_size}")
print(f"biggest single loss: {round(executed_trades_df['profit'].min(), 2)} {bot.account.currency}")
print(f"biggest single win: {round(executed_trades_df['profit'].max(), 2)} {bot.account.currency}")
print(f"initial balance: {round(inital_balance, 2)} {bot.account.currency}")
print(f"account balance: {round(account_balance, 2)} {bot.account.currency}")
print(f"max drawdown: {round(inital_balance - max_drawdown, 2)} {bot.account.currency}")
print(f"net profit: {round(gross_profit + loss, 2)} {bot.account.currency}")

# Create the subplots with 2 rows
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[0.8, 0.2],
                    subplot_titles=('Candlestick Chart', 'MACD Line'))

# Add candlestick chart to the first subplot
fig.add_trace(go.Candlestick(x=df['time'],
                             open=df['open'],
                             high=df['high'],
                             low=df['low'],
                             close=df['close'],
                             name='Candlestick'), row=1, col=1)

# Add buy signals (up arrows) to the first subplot
fig.add_trace(go.Scatter(
    x=executed_trades_df[executed_trades_df['is_buy2'] == True]['time'],
    y=executed_trades_df[executed_trades_df['is_buy2'] == True]['low'] * 0.999,
    mode='markers',
    marker=dict(symbol='arrow-up', color='green', size=10),
    name='Buy Signal'
), row=1, col=1)

# Add sell signals (down arrows) to the first subplot
fig.add_trace(go.Scatter(
    x=executed_trades_df[executed_trades_df['is_sell2'] == True]['time'],
    y=executed_trades_df[executed_trades_df['is_sell2'] == True]['high'] * 1.001,
    mode='markers',
    marker=dict(symbol='arrow-down', color='red', size=10),
    name='Sell Signal'
), row=1, col=1)

# Draw buy and sell orders on the chart

for index, row in executed_trades_df.iterrows():
    if not (row['successful']):
        continue
    fig.add_shape(
        type="rect",
        x0=row['time'], x1=row['position_close_time'],
        y0=row['close'], y1=row['tp'],
        line=dict(color="green", width=2),
        fillcolor="green",
        opacity=0.3,
        row=1, col=1
    )
    fig.add_shape(
        type="rect",
        x0=row['time'], x1=row['position_close_time'],
        y0=row['sl'], y1=row['close'],
        line=dict(color="red", width=2),
        fillcolor="red",
        opacity=0.3,
        row=1, col=1
    )


# Add LMSA Upper Band line to the first subplot
fig.add_trace(go.Scatter(x=df['time'], 
                         y=df['lsma_upper_band'], 
                         mode='lines', 
                         name='LMSA Upper Band'
                         ), row=1, col=1)

# Add LMSA Lower Band line to the first subplot
fig.add_trace(go.Scatter(x=df['time'],
                         y=df['lsma_lower_band'], 
                         mode='lines', 
                         name='LMSA Lower Band'
                         ), row=1, col=1)

# Add LMSA Band line to the first subplot
fig.add_trace(go.Scatter(x=df['time'], y=df['lsma'], 
                         mode='lines', name='LMSA'), row=1, col=1)

# Add MACD Line to the second subplot
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['macd_line'],
    name='MACD Line',
    line=dict(color='purple')
), row=2, col=1)

# Add MACDs Line to the second subplot
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['macd_signal'],
    name='MACD Signal',
    line=dict(color='blue')
), row=2, col=1)

# Update layout
fig.update_layout(title='XAUUSD',
                  #xaxis_title='Date',
                  #yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  template="plotly_dark"
                  )

fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
    ]
)



# Show the plot
fig.show()

del relevant_ticks
del filtered_df
del df
#nothing

