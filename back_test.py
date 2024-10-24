import  MetaTrader5 as mt5 
import pandas as pd
from strategy import *
from analysis import *
from main import bot

account_balance = 700
inital_balance = account_balance


start = pd.Timestamp("2024-04-01")
end = pd.Timestamp("2024-04-30")


df = apply_strategy(start=start, end=end)
filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()

if not filtered_df.empty:
    results = analyse(filtered_df=filtered_df,
            symbol=bot.symbol,
            bot=bot,
            account_balance=account_balance,
            lot_size=bot.lot,
            timeframe=bot.timeframe)

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
    running_trades = results['running_trades']
    count_auto = (executed_trades_df['exit'] == 'auto').sum()
    count_manual = (executed_trades_df['exit'] == 'manual').sum()
    print(f"\nanalysis from {start} to {end}\n")
    print(f"\nPROFITABILITY\n")

    print(f"lot size used: {bot.lot}")
    print(f"Total trades: {total_trades}")
    print(f"total auto close: {count_auto}")
    print(f"total manual close: {count_manual}")
    print(f"Total unexecuted trades: {unexecuted_trades}")
    print(f"Total successful trades: {successful_trades}") 
    print(f"Total unsuccessful trades: {unsuccessful_trades}")
    print(f"Total break even trades: {break_even}")
    print(f"Total running trades: {running_trades}")
    print(f"gross profit: {round(gross_profit, 2)} {bot.account.currency}")
    print(f"loss: {round(loss, 2)} {bot.account.currency}")
    print(f"percentage profitability: {percentage_profitability} %")
    print(f"max win streak: {executed_trades_df['win_streak'].max()}")
    print(f"max loosing streak: {executed_trades_df['losing_streak'].max()}")



    print(f"profit factor: {round(profit_factor, 2)}")
    print(f"\nACCOUNT DETAILS\n")
    min_profit = executed_trades_df['profit'].min()
    adjusted_min_profit = min_profit if min_profit < 0 else 0
    print(f"biggest single loss: {round(adjusted_min_profit, 2)} {bot.account.currency}")
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

filtered_df.to_csv('csv/filtered_df.csv', index=False)
# bot.get_ticks(symbol=symbol,start=start,end=end).to_csv("csv/ticks.csv", index=False)
