from datetime import datetime
import  MetaTrader5 as mt5 
from beta_trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd

from beta_strategy import *
import numpy as np
from analysis import *

def auto_trendline(data):
    data['time'] = data['time'].astype('datetime64[s]')
    data = data.set_index('time',drop=False)

    # Take natural log of data to resolve price scaling issues
    df_log = np.log(data[['high', 'low', 'close']])

    # Trendline parameter
    lookback = 15

    # Initialize columns for trendlines and their gradients
    data['support_trendline'] = np.nan
    data['resistance_trendline'] = np.nan
    data['support_gradient'] = np.nan
    data['resistance_gradient'] = np.nan

    # Lists to hold the trendline segments
    support_trendline_x = []
    support_trendline_y = []
    resistance_trendline_x = []
    resistance_trendline_y = []

    # Iterate from the end of the dataset towards the beginning
    for i in range(len(df_log) - 1, lookback - 2, -1):
        window_data = df_log.iloc[i - lookback + 1: i + 1]
        support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
        
        # Calculate trendline values and gradients for each point in the window
        for j in range(lookback):
            idx = i - lookback + 1 + j
            support_value = support_coefs[0] * j + support_coefs[1]
            resist_value = resist_coefs[0] * j + resist_coefs[1]
            data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
            data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
            data.at[data.index[idx], 'support_gradient'] = support_coefs[0]
            data.at[data.index[idx], 'resistance_gradient'] = resist_coefs[0]
            
            # Append to trendline segments
            support_trendline_x.append(data.index[idx])
            support_trendline_y.append(np.exp(support_value))
            resistance_trendline_x.append(data.index[idx])
            resistance_trendline_y.append(np.exp(resist_value))
        
        # Append None to break the line
        support_trendline_x.append(None)
        support_trendline_y.append(None)
        resistance_trendline_x.append(None)
        resistance_trendline_y.append(None)

    

    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])

    # Add support and resistance lines
    fig.add_trace(go.Scatter(
        x=support_trendline_x,
        y=support_trendline_y,
        mode='lines',
        name='Support Line',
        line=dict(color='green'),
        connectgaps=False,
        line_shape='linear'
    ))

    fig.add_trace(go.Scatter(
        x=resistance_trendline_x,
        y=resistance_trendline_y,
        mode='lines',
        name='Resistance Line',
        line=dict(color='red'),
        connectgaps=False,
        line_shape='linear'
    ))

    # Update layout
    fig.update_layout(
        title='Candlestick Chart with Support and Resistance Lines',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    # Show the figure
    fig.show()
    return data


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
start = pd.to_datetime(datetime(2024,5,4))
conversion = timeframe_to_interval.get(timeframe, 3600)
end = pd.to_datetime(datetime(2024,5,14))   #(pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

#creating dataframe by importing trade data
data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)

data=auto_trendline(data)


df = m15_gold_strategy(data.copy())

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()

total_trades = 0
unexecuted_trades = 0
successful_trades = 0
unsuccessful_trades = 0
gross_profit = 0
loss = 0
executed_trades = [] 
biggest_loss = 0
biggest_win = 0

inital_balance = account_balance
break_even = 0

for index, row in filtered_df.iterrows():
    #if trade is in valid, no further processing
    
    if check_invalid_stopouts(row):
        unexecuted_trades += 1
        print(f"trade invalid stopouts: {row['time']}")
        continue

    # allways add 15 min to start time because position was started at cnadle close and not open
    start_time= (row['time'] + pd.Timedelta(seconds=1)).ceil(conversion) #add 1 second to be able to apply ceil function
    end_date =  pd.Timestamp.now() + pd.Timedelta(hours=1)    
   
    #fetch data to compare stop levels and see which was reached first, trailing stop is calculated only after every candle close
    relevant_ticks = bot.get_ticks(symbol=symbol,start=start_time,end=end_date)
    second_chart = bot.chart(symbol=symbol, timeframe=timeframe, start=start_time, end=end_date)
    
    # Check if stop loss or take profit or trailing stop was reached 
    stop_loss_reached = (relevant_ticks['bid'] <= row["sl"]) if row["is_buy2"] else (relevant_ticks['bid'] >= row["sl"])
    take_profit_reached = (relevant_ticks['bid'] >= row["tp"]) if row["is_buy2"] else (relevant_ticks['bid'] <= row["tp"])
    trailing_stop_reached = (second_chart['close'] >= row["be_condition"]) if row["is_buy2"] else (second_chart['close'] <= row["be_condition"])

    #find the index at which it each level was reached... argmax will return the first occurence of this
    trailing_stop_index = np.argmax(trailing_stop_reached) if trailing_stop_reached.any() else -1
    stop_loss_index = np.argmax(stop_loss_reached) if stop_loss_reached.any() else -1
    take_profit_index = np.argmax(take_profit_reached) if take_profit_reached.any() else -1
    
    #find the corresponding time to the indexes
    time_to_trail = (second_chart.loc[trailing_stop_index, "time"] + pd.Timedelta(seconds=1)).ceil(conversion) if trailing_stop_index != -1 else pd.Timestamp.max
    time_tp_hit = relevant_ticks.loc[take_profit_index, 'time'] if take_profit_index != -1 else pd.Timestamp.max
    time_sl_hit = relevant_ticks.loc[stop_loss_index, 'time'] if stop_loss_index != -1 else pd.Timestamp.max
    
    #trail stop loss if needed
    row['sl_updated'] = True if min(time_sl_hit, time_tp_hit, time_to_trail) == time_to_trail else False
    row['time_updated'] = time_to_trail if min(time_sl_hit, time_tp_hit, time_to_trail) == time_to_trail else None

    #update actual sl and refind teh indexes
    if  row['sl_updated']:
        relevant_ticks = bot.get_ticks(symbol=symbol,start=time_to_trail,end=end_date) # Filter ticks dataframe from time_value onwards
        
        #update the stop loss level
        row['sl'] = row['be']
        stop_loss_reached = relevant_ticks['bid'] <= row['sl'] if row['is_buy2'] else relevant_ticks['bid'] >= row['sl']

        #find the new time for ehich stop loss was hit
        stop_loss_index = np.argmax(stop_loss_reached) if stop_loss_reached.any() else -1
        time_sl_hit = relevant_ticks.loc[stop_loss_index, 'time'] if stop_loss_index != -1 else pd.Timestamp.max
    
    #save final updated times
    row['time_to_trail']  = time_to_trail
    row['time_tp_hit']  = time_tp_hit
    row['time_sl_hit']  = time_sl_hit
    print(f"Currently Working on Trade: {row['time']} where sl update is: {row['sl_updated']}")
    print(f"tp time: {time_tp_hit}")
    print(f"sl time: {time_sl_hit}")
    print(f"tr time: {time_to_trail}")

    filtered_ticks = relevant_ticks[
        (relevant_ticks['time'] >= (time_to_trail if row['sl_updated'] else start_time)) & 
        (relevant_ticks['time'] <= min(time_sl_hit, time_tp_hit))
    ].copy()
    
    max_min = filtered_ticks['bid'].max() if row['is_buy2'] else filtered_ticks['bid'].min()
    row['max_completion'] = calculate_percentage_completion(entry_price=row['close'], goal_price=row['tp'], current_price=max_min, is_buy=row['is_buy2'])
    row['max_floating_profit'] = bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=max_min) \
                                    if row['is_buy2'] else \
                                    bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=max_min)        

    if stop_loss_index == 0 or take_profit_index == 0:
        print(f"take profit or stop loss reached ar zero for trade {row['time']}")
        unexecuted_trades +=1
        continue
    total_trades+=1
    executed_trades.append(row)

    if stop_loss_reached.any() and take_profit_reached.any():
        if(min(time_sl_hit, time_tp_hit) == time_tp_hit):
            print("trade successful")
            row['type'] = "success"
            row['success'] = True
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
        
        elif(row['sl_updated']):       
            row['type'] = "even"
            print("trade broke even")
            row['success'] = True
            break_even +=1
            if row["is_buy2"]:
                row['profit'] =  bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["sl"])
                gross_profit += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
            else: 
                row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["sl"])        
                gross_profit +=  row['profit'] 
                account_balance  += row['profit']
                row["account_balance"] = account_balance

        else:
            print("trade failed")
            row['type'] = "fail"
            row['success'] = False
            unsuccessful_trades +=1            
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
        if(row['sl_updated']):
            print("trade broke even")
            row['type'] = "even"
            row['success'] = True
            break_even +=1
            if row["is_buy2"]:
                row['profit'] =  bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["sl"])
                gross_profit += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
            else: 
                row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["sl"])        
                gross_profit +=  row['profit'] 
                account_balance  += row['profit']
                row["account_balance"] = account_balance
        else:
            print("trade failed")
            row['type'] = "fail"
            row['success'] = False
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
        print("trade successful")
        successful_trades+=1
        row['type'] = "success"
        row['success'] = True

        if row["is_buy2"]:
                row['profit'] =  bot.cal_profit(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, lot=lot_size, open_price=row["close"], close_price=row["sl"])
                gross_profit += row['profit']
                account_balance  += row['profit']
                row["account_balance"] = account_balance
        else: 
            row['profit'] = bot.cal_profit(symbol=symbol,order_type=mt5.ORDER_TYPE_SELL,lot=lot_size, open_price=row["close"], close_price=row["sl"])        
            gross_profit +=  row['profit'] 
            account_balance  += row['profit']
            row["account_balance"] = account_balance
    else:
        row['type'] = "running"
        row['success'] = False
        row["account_balance"] = account_balance
        row['profit'] = 0
        row['position_close_time'] = row['time'] + pd.Timedelta(hours=3)
        print(f"Neither stop loss nor take profit was reached for trade. {row["time"]}")

    

if loss != 0:
    profit_factor = gross_profit / abs(loss)
else:
    profit_factor = float('inf')  # Handle case where there are no losing trades

if total_trades > 0:
    percentage_profitability = ((successful_trades+break_even) / (total_trades)) * 100
else:
    percentage_profitability = 0  # Handle case where there are no trades


executed_trades_df = pd.DataFrame(executed_trades)


grouper = (executed_trades_df['success'] != executed_trades_df['success'].shift()).cumsum()
executed_trades_df['win_streak'] = executed_trades_df.groupby(grouper)['success'].transform('cumsum')
# Calculate losing streak
executed_trades_df['losing_streak'] = (
    ~executed_trades_df['success'].copy()  # Invert a copy of the success column
    ).groupby(grouper).cumsum()


df.to_csv('beta/output.csv', index=False)
filtered_df.to_csv('beta/filtered_df.csv', index=False)
executed_trades_df.to_excel('filtered_excel_df.xlsx')
executed_trades_df.to_csv('beta/executed_trades_df.csv', index=False)
bot.get_ticks(symbol=symbol,start=start,end=end_date).to_csv("beta/ticks.csv", index=False)

print(f"\nanalysis from {start} to {end}\n")
print(f"\nPROFITABILITY\n")

print(f"lot size used: {lot_size}")
print(f"Total trades: {total_trades}")
print(f"Total unexecuted trades: {unexecuted_trades}")
print(f"Total successful trades: {successful_trades}") 
print(f"Total unsuccessful trades: {unsuccessful_trades}")
print(f"Total break even trades: {break_even}")
print(f"gross profit: {round(gross_profit, 2)} {bot.account.currency}")
print(f"loss: {round(loss, 2)} {bot.account.currency}")
print(f"percentage profitability: {percentage_profitability} %")
print(f'max win streak: {executed_trades_df['win_streak'].max()}')
print(f'max loosing streak: {executed_trades_df['losing_streak'].max()}')
print(f'avg maximum order fill: {executed_trades_df['max_completion'].mean()} %')
print(f'lowest order fill: {executed_trades_df['max_completion'].min()} %')



print(f"profit factor: {round(profit_factor, 2)}")
print(f"\nACCOUNT DETAILS\n")
print(f"lot size used: {lot_size}")
print(f"biggest single loss: {round(executed_trades_df['profit'].min(), 2)} {bot.account.currency}")
print(f"biggest single win: {round(executed_trades_df['profit'].max(), 2)} {bot.account.currency}")
print(f"initial balance: {round(inital_balance, 2)} {bot.account.currency}")
print(f"account balance: {round(account_balance, 2)} {bot.account.currency}")
print(f"lowest account balance: {round(executed_trades_df['account_balance'].min(), 2)} {bot.account.currency}")
print(f"net profit: {round(gross_profit + loss, 2)} {bot.account.currency}")




# Show the plot
display_chart(df)

del relevant_ticks
del filtered_df
del df
#nothing

