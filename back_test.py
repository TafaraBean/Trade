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
    lookback = 30

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
inital_balance = account_balance
lot_size = 0.01
timeframe = mt5.TIMEFRAME_M15

conversion = bot.timeframe_to_interval.get(timeframe, 3600)
start = pd.Timestamp("2024-04-10")
#end = pd.Timestamp("2024-03-20 23:00:00")   
end = (pd.Timestamp.now() + pd.Timedelta(hours=1)).floor(conversion)

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
    print(f"\nweekly profit:\n {weekly_df}")
    print(f"\nmonthly profit:\n {monthly_df}")

    executed_trades_df.to_excel('filtered_excel_df.xlsx')
    executed_trades_df.to_csv('csv/executed_trades_df.csv', index=False)
else:
    print("No Buy or Sell signals were generated using this strategy")


display_chart(df)

merged_data.to_csv('csv/merged_data.csv', index=False)
filtered_df.to_csv('csv/filtered_df.csv', index=False)
# bot.get_ticks(symbol=symbol,start=start,end=end).to_csv("csv/ticks.csv", index=False)
