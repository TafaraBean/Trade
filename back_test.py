import  MetaTrader5 as mt5 
import pandas as pd
from strategy import *
from analysis import *
from main import bot

account_balance = 700
inital_balance = account_balance


#start = pd.Timestamp("2024-12-05 00:00:01")
end = pd.Timestamp('2024-12-05')
start = end - pd.Timedelta(days=1)
# start = end - pd.Timedelta(days=4)

data = bot.copy_chart_range(symbol=bot.symbol,
                            timeframe=bot.timeframe,
                            start=start,
                            end=end)

df = apply_strategy(df=data)

filtered_df = df[(df['is_buy2'] == True) | (df['is_sell2'] == True)].copy()

if not filtered_df.empty:
    results = analyse(filtered_df=filtered_df,
            bot=bot,
            close_opp_trades= False,
            account_balance=account_balance,
            timeframe=bot.timeframe)

    executed_trades_df = pd.DataFrame(results['executed_trades_df'])
    summary_df = executed_trades_df[['time','open','high','low','close','profit','entry_price','exit_price','type','account_balance','win_streak','losing_streak']]
    weekly_df = pd.DataFrame(results['weekly_profit'])
    monthly_df = pd.DataFrame(results['monthly_profit'])


    count_auto = (executed_trades_df['exit'] == 'auto').sum()
    count_manual = (executed_trades_df['exit'] == 'manual').sum()

    print(f"\nanalysis from {start} to {end}\n")
    print(f"\nPROFITABILITY\n")

    print(f"lot size used: {bot.lot}")
    print(f"Total trades: {results['total_trades']}")
    print(f"total auto close: {count_auto}")
    print(f"total manual close: {count_manual}")
    print(f"Total unexecuted trades: {results['unexecuted_trades']}")
    print(f"Total successful trades: {results['successful_trades']}") 
    print(f"Total unsuccessful trades: {results['unsuccessful_trades']}")
    print(f"Total break even trades: {results['break_even']}")
    print(f"Total running trades: {results['running_trades']}")
    print(f"gross profit: {round(results['gross_profit'], 2)} {bot.account.currency}")
    print(f"loss: {round(results['loss'], 2)} {bot.account.currency}")
    print(f"percentage profitability: {results['percentage_profitability']} %")
    print(f"max win streak: {executed_trades_df['win_streak'].max()}")
    print(f"max loosing streak: {executed_trades_df['losing_streak'].max()}")



    print(f"profit factor: {round(results['profit_factor'], 2)}")
    print(f"\nACCOUNT DETAILS\n")
    min_profit = executed_trades_df['profit'].min()
    adjusted_min_profit = min_profit if min_profit < 0 else 0
    print(f"biggest single loss: {round(adjusted_min_profit, 2)} {bot.account.currency}")
    print(f"biggest single win: {round(executed_trades_df['profit'].max(), 2)} {bot.account.currency}")
    print(f"initial balance: {round(inital_balance, 2)} {bot.account.currency}")
    print(f"account balance: {round(results['account_balance'], 2)} {bot.account.currency}")
    print(f"lowest account balance: {round(executed_trades_df['account_balance'].min(), 2)} {bot.account.currency}")
    print(f"net profit: {round(results['gross_profit'] + results['loss'], 2)} {bot.account.currency}")

    # Print the DataFrame with monthly profit (optional)
    print(f"\nweekly profit:\n{weekly_df.to_string(index=False)}")
    print(f"\nmonthly profit:\n{monthly_df.to_string(index=False)}")

    executed_trades_df.to_excel('filtered_excel_df.xlsx')
    executed_trades_df.to_csv('csv/executed_trades_df.csv', index=False)
    summary_df.to_csv('csv/summary.csv', index=False)
else:
    print("No Buy or Sell signals were generated using this strategy")


display_chart(df)

filtered_df.to_csv('csv/filtered_df.csv', index=False)
# bot.get_ticks(symbol=symbol,start=start,end=end).to_csv("csv/ticks.csv", index=False)
