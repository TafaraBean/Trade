import  MetaTrader5 as mt5 
from trading_bot import TradingBot
import os
import pandas as pd
import pandas_ta as ta

from strategy import *
import numpy as np
from analysis import *

def auto_trendline(data):
    print("appling auto trendline...")
    data['time2'] = data['time'].astype('datetime64[s]')
    data = data.set_index('time', drop=True)
    print("hourly data:")
    print(data)
    print("==========")

    # Take natural log of data to resolve price scaling issues
    df_log = np.log(data[['high', 'low', 'close']])

    # Trendline parameter
    lookback = 8

    # Initialize columns for trendlines and their gradients
    data['support_trendline'] = np.nan
    data['resistance_trendline'] = np.nan
    data['support_gradient'] = np.nan
    data['resistance_gradient'] = np.nan
    data['hour_lsma'] = ta.linreg(data['close'], length=8)
    data['prev_hour_lsma']=data['hour_lsma'].shift(1)
    data['hour_lsma_slope'] = data['hour_lsma'].diff()
    data['prev_hour_lsma_slope']= data['hour_lsma_slope'].shift(1)
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data['hour_macd_line'] = macd['MACD_12_26_9']
    data['prev_hour_macd_line']=data['hour_macd_line'].shift(1)


    # Iterate over the dataset in overlapping windows of 15 candles
    for i in range(lookback, len(df_log) + 1):
        current_index = df_log.index[i-1]
        window_data = df_log.iloc[i - lookback:i]
        support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
        
        # Extract slope and intercept
        support_slope, support_intercept = support_coefs
        resist_slope, resist_intercept = resist_coefs
        data.at[current_index, 'fixed_resistance_gradient'] = resist_slope
        data.at[current_index, 'fixed_support_gradient'] = support_slope
        support_value = support_slope * window_data.at[current_index,'low'] + support_intercept
        resist_value = resist_slope * window_data.at[current_index,'high'] + resist_intercept
        data.at[current_index, 'fixed_support_trendline'] = np.exp(support_value)
        data.at[current_index, 'fixed_resistance_trendline'] = np.exp(resist_value)
        # Apply the calculated gradients to each candle in the window
        
        for j in range(lookback):
            idx = i - lookback + j
            support_value = support_slope * j + support_intercept
            resist_value = resist_slope * j + resist_intercept
            data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
            data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
            data.at[data.index[idx], 'support_gradient'] = support_slope
            data.at[data.index[idx], 'resistance_gradient'] = resist_slope

    data.to_csv("csv/test.csv",index=False)
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
        x=data.index,
        y=data['support_trendline'],
        mode='lines',
        name='Support Line',
        line=dict(color='green'),
        connectgaps=False
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['resistance_trendline'],
        mode='lines',
        name='Resistance Line',
        line=dict(color='red'),
        connectgaps=False
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


account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="BTCUSD"
account_balance = 450
lot_size = 0.01
timeframe = mt5.TIMEFRAME_M15
<<<<<<< HEAD:beta/trade.py
start = pd.Timestamp("2024-03-10")
conversion = timeframe_to_interval.get(timeframe, 3600)
end = pd.Timestamp("2024-03-20 23:00:00")   
=======
conversion = bot.timeframe_to_interval.get(timeframe, 3600)

start = pd.Timestamp("2024-05-29")
end = pd.Timestamp("2024-05-30 23:00:00")
>>>>>>> 3658e7a25e8d52a967ba1957f874002ff731cfc6:back_test.py
#end = (pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

#creating dataframe by importing trade data
data = bot.copy_chart_range(symbol=symbol, timeframe=timeframe, start=start, end=end)
hour_data = bot.copy_chart_range(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)



hour_data=auto_trendline(hour_data)
hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline']]

hour_data.to_csv("csv/hour_data.csv",index=False)

data['hourly_time']=data['time'].dt.floor('h')

merged_data = pd.merge(data,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))


df = m15_gold_strategy(merged_data)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()

total_trades = 0
unexecuted_trades = 0
successful_trades = 0
unsuccessful_trades = 0
gross_profit = 0
loss = 0
executed_trades = [] 

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
    end_time= end + pd.Timedelta(hours=1).ceil(conversion)
    #fetch data to compare stop levels and see which was reached first, trailing stop is calculated only after every candle close
    relevant_ticks = bot.get_ticks_range(symbol=symbol,start=start_time,end=end_time)
    second_chart = bot.copy_chart_range(symbol=symbol, timeframe=timeframe, start=start_time, end=end_time)
    
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
    
    #trail stop loss if needed, 
    #also factor in probability of trading still running and none of the levels were ever reached
    if time_to_trail == pd.Timestamp.max and time_tp_hit == pd.Timestamp.max and time_sl_hit == pd.Timestamp.max:
        row['sl_updated'] = False
    else:
        row['sl_updated'] = min(time_sl_hit, time_tp_hit, time_to_trail) == time_to_trail
    
    row['time_updated'] = time_to_trail if min(time_sl_hit, time_tp_hit, time_to_trail) == time_to_trail else None

    #update actual sl and refind teh indexes
    if  row['sl_updated']:
        relevant_ticks = bot.get_ticks_range(symbol=symbol,start=time_to_trail,end=end_time) # Filter ticks dataframe from time_value onwards
        
        #update the stop loss level
        row['sl'] = row['be']
        stop_loss_reached = relevant_ticks['bid'] <= row['sl'] if row['is_buy2'] else relevant_ticks['bid'] >= row['sl']

        #find the new time for ehich stop loss was hit
        stop_loss_index = np.argmax(stop_loss_reached) if stop_loss_reached.any() else -1
        time_sl_hit = relevant_ticks.loc[stop_loss_index, 'time'] if stop_loss_index != -1 else pd.Timestamp.max
    
    #save final updated times
    row['time_to_trail'] = None if time_to_trail == pd.Timestamp.max else time_to_trail
    row['time_tp_hit'] = None if time_tp_hit == pd.Timestamp.max else time_tp_hit
    row['time_sl_hit'] = None if time_sl_hit == pd.Timestamp.max else time_sl_hit

    print(f"Currently Working on Trade: {row['time']} where sl update is: {row['sl_updated']}")
    print(f"tp time: {row['time_tp_hit']}")
    print(f"sl time: {row['time_sl_hit'] }")
    print(f"tr time: {row['time_to_trail']}")

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

    if stop_loss_index > -1 and take_profit_index > -1:
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
    elif take_profit_index == -1 and stop_loss_index != -1:
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
    elif stop_loss_index == -1 and take_profit_index != -1:
        print("trade successful")
        successful_trades+=1
        row['type'] = "success"
        row['success'] = True

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

# Set the 'Week' column based on the year and week number
executed_trades_df['Week'] = executed_trades_df['time'].dt.isocalendar().week

# Calculate weekly profit by grouping by 'Week' and summing 'profit'
weekly_profit = executed_trades_df.groupby('Week')['profit'].sum()

# Create a new DataFrame with 'Week' and 'Weekly Profit' columns
weekly_df = pd.DataFrame({'Week': weekly_profit.index, 'Weekly Profit': weekly_profit.values})


# Set the 'Month' column based on the year and month number
executed_trades_df['Month'] = executed_trades_df['time'].dt.month

# Calculate monthly profit by grouping by 'Month' and summing 'profit'
monthly_profit = executed_trades_df.groupby('Month')['profit'].sum()

# Create a new DataFrame with 'Month' and 'Monthly Profit' columns (optional)
monthly_df = pd.DataFrame({'Month': monthly_profit.index, 'Monthly Profit': monthly_profit.values})

# Print the DataFrame with monthly profit (optional)
print(f"\nweekly profit:\n {weekly_df}")


print(f"\nmonthly profit:\n {monthly_df}")


df.to_csv('csv/output.csv', index=False)
# filtered_df.to_csv('csv/filtered_df.csv', index=False)
executed_trades_df.to_excel('filtered_excel_df.xlsx')
executed_trades_df.to_csv('csv/executed_trades_df.csv', index=False)
# bot.get_ticks(symbol=symbol,start=start,end=end).to_csv("csv/ticks.csv", index=False)

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

