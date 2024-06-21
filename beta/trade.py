from datetime import datetime
import  MetaTrader5 as mt5 
from beta_trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd

from beta_strategy import *
import numpy as np
from analysis import *


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
start = pd.to_datetime(datetime(2024,4,1))
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
    
    
    if stop_loss_index == 0 or take_profit_index == 0:
        print(f"take profit or stop loss reached ar zero for trade {row['time']}")
        unexecuted_trades +=1
        continue
    total_trades+=1
    executed_trades.append(row)

    if stop_loss_reached.any() and take_profit_reached.any():
        if(min(time_sl_hit, time_tp_hit) == time_tp_hit):
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


data.to_csv('beta/output.csv', index=False)
filtered_df.to_csv('beta/filtered_df.csv', index=False)
executed_trades_df.to_excel('filtered_excel_df.xlsx')
executed_trades_df.to_csv('beta/executed_trades_df.csv', index=False)
#bot.get_ticks(symbol=symbol,start=start,end=end_date).to_csv("beta/ticks.csv", index=False)

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
print(f'win streak: {executed_trades_df['win_streak'].max()}')
print(f'loosing streak: {executed_trades_df['losing_streak'].max()}')

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
#display_chart(df)

del relevant_ticks
del filtered_df
del df
#nothing

