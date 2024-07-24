import  MetaTrader5 as mt5 
from trading_bot import TradingBot
import os
import pandas as pd
from dotenv import load_dotenv
from strategy import *
from analysis import *

load_dotenv()

account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( login=account, password=password, server=server)
symbol="EURUSD.Z"
account_balance = 700
inital_balance = account_balance
lot_size = 0.01
timeframe = mt5.TIMEFRAME_M15

conversion = bot.timeframe_to_interval.get(timeframe, 3600)
start = pd.Timestamp("2024-07-01")
#end = pd.Timestamp("2024-01-28")
end = (pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

#creating dataframe by importing trade data
data = bot.copy_chart_range(symbol=symbol, timeframe=timeframe, start=start, end=end)
data=auto_trendline_15(data)

hour_data = bot.copy_chart_range(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)



hour_data=auto_trendline(hour_data)

hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline','prev_fixed_support_trendline','prev_fixed_resistance_trendline','prev_fixed_resistance_gradient','prev_fixed_support_gradient','ema_50','ema_24','stoch_k','stoch_d','prev_hour_macd_signal','prev_psar','prev_psar_direction','prev_nadaraya_watson','prev_nadaraya_watson_trend']]

hour_data.to_csv("csv/hour_data.csv",index=False)

data['hourly_time']=data['time'].dt.floor('h')

merged_data = pd.merge(data,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))


df = m15_gold_strategy(merged_data)
filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()



if not filtered_df.empty:
    results = analyse(filtered_df=filtered_df,
            symbol=symbol,
            bot=bot,
            account_balance=account_balance,
            lot_size=lot_size,
            timeframe=timeframe)

    executed_trades_df = pd.DataFrame(results['executed_trades_df'])
    weekly_df = pd.DataFrame(results['weekly_profit'])
    monthly_df = pd.DataFrame(results['monthly_profit'])
    percentage_profitability= results['percentage_profitability']
    profit_factor =results['profit_factor']
    total_trades = results['total_trades']
    unexecuted_trades = results['unexecuted_trades']
    unsuccessful_trades = results['unsuccessful_trades']
    successful_trades = results['successful_trades']
    break_even = results['break_even']
    profit_factor = results['profit_factor']
    gross_profit = results['gross_profit']
    loss = results['loss']
    account_balance = results['account_balance']


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
    print(f"max win streak: {executed_trades_df['win_streak'].max()}")
    print(f"max loosing streak: {executed_trades_df['losing_streak'].max()}")



    print(f"profit factor: {round(profit_factor, 2)}")
    print(f"\nACCOUNT DETAILS\n")
    print(f"lot size used: {lot_size}")
    print(f"biggest single loss: {round(executed_trades_df['profit'].min(), 2)} {bot.account.currency}")
    print(f"biggest single win: {round(executed_trades_df['profit'].max(), 2)} {bot.account.currency}")
    print(f"initial balance: {round(inital_balance, 2)} {bot.account.currency}")
    print(f"account balance: {round(account_balance, 2)} {bot.account.currency}")
    print(f"lowest account balance: {round(executed_trades_df['account_balance'].min(), 2)} {bot.account.currency}")
    print(f"net profit: {round(gross_profit + loss, 2)} {bot.account.currency}")

    # Print the DataFrame with monthly profit (optional)
    print(f"\nweekly profit:\n{weekly_df.to_string(index=False)}")
    print(f"\nmonthly profit:\n{monthly_df.to_string(index=False)}")

    executed_trades_df.to_excel('filtered_excel_df.xlsx')
    executed_trades_df.to_csv('csv/executed_trades_df.csv', index=False)
else:
    print("No Buy or Sell signals were generated using this strategy")


display_chart(df)

merged_data.to_csv('csv/merged_data.csv', index=False)
filtered_df.to_csv('csv/filtered_df.csv', index=False)
# bot.get_ticks(symbol=symbol,start=start,end=end).to_csv("csv/ticks.csv", index=False)
