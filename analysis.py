import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_bot import TradingBot
import numpy as np
from typing import Tuple

def check_invalid_stopouts(row):
    """
    Efficiently checks for invalid stopouts in a trade row.

    Args:
        row (pd.Series): A row from the filtered DataFrame containing trade information.

    Returns:
        bool: True if the trade has invalid stopouts, False otherwise.
    """

    is_buy2 = row['is_buy2']
    close_price = row["close"]
    take_profit = row["tp"]
    stop_loss = row["sl"]

    # Consolidated condition for invalid stopouts based on buy/sell direction
    invalid_stopouts = (is_buy2 and (take_profit <= close_price or stop_loss >= close_price)) or \
                       (not is_buy2 and (take_profit >= close_price or stop_loss <= close_price))

    return invalid_stopouts


def display_chart(df):
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
        x=df[df['is_buy2'] == True]['time'],
        y=df[df['is_buy2'] == True]['low'] * 0.999,
        mode='markers',
        marker=dict(symbol='arrow-up', color='green', size=10),
        name='Buy Signal'
    ), row=1, col=1)

    # Add sell signals (down arrows) to the first subplot
    fig.add_trace(go.Scatter(
        x=df[df['is_sell2'] == True]['time'],
        y=df[df['is_sell2'] == True]['high'] * 1.001,
        mode='markers',
        marker=dict(symbol='arrow-down', color='red', size=10),
        name='Sell Signal'
    ), row=1, col=1)



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
    
    fig.show()


def calculate_percentage_completion(entry_price, goal_price, current_price, is_buy):
  """
  Calculates the percentage completion of a running trade.

  Args:
      entry_price (float): The price at which the trade was entered.
      goal_price (float): The target price for the trade.
      current_price (float): The current market price of the asset.
      is_buy (bool): True if it's a buy trade, False if it's a sell trade.

  Returns:
      float: The percentage completion of the trade (0.0 to 1.0).
  """

  if entry_price == goal_price:
    return 1.0  # Handle case where entry and goal prices are equal

  # Calculate the price movement direction
  price_movement = current_price - entry_price

  # Adjust calculation based on buy or sell trade
  if is_buy:
    completion_ratio = price_movement / (goal_price - entry_price)
  else:
    completion_ratio = (entry_price - current_price) / (entry_price - goal_price)

  # Clamp the completion ratio to the range [0.0, 1.0]
  completion_ratio = max(0.0, min(completion_ratio, 1.0))

  return completion_ratio * 100  # Convert to percentage and return

def calc_weekly_proft(executed_trades_df):
    executed_trades_df['Week'] = executed_trades_df['time'].dt.isocalendar().week

    # Calculate weekly profit by grouping by 'Week' and summing 'profit'
    weekly_profit = executed_trades_df.groupby('Week')['profit'].sum()

    # Create a new DataFrame with 'Week' and 'Weekly Profit' columns
    weekly_df = pd.DataFrame({'Week': weekly_profit.index, 'Weekly Profit': weekly_profit.values})
    return weekly_profit

def calc_monthly_proft(executed_trades_df):
    # Set the 'Month' column based on the year and month number
    executed_trades_df['Month'] = executed_trades_df['time'].dt.month

    # Calculate monthly profit by grouping by 'Month' and summing 'profit'
    monthly_profit = executed_trades_df.groupby('Month')['profit'].sum()
    monthly_df = pd.DataFrame({'Month': monthly_profit.index, 'Monthly Profit': monthly_profit.values})

    return monthly_df

import pandas as pd
import numpy as np


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope

    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err;


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y.iloc[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)











'''
# Plot Trendlines on candles 
# Library for plotting candles
# pip install mplfinance
import mplfinance as mpf 



candles = data.iloc[-30:] # Last 30 candles in data
support_coefs_c, resist_coefs_c = fit_trendlines_single(candles['close'])
support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])

support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]

support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

plt.style.use('dark_background')
ax = plt.gca()

def get_line_points(candles, line_points):
    # Place line points in tuples for matplotlib finance
    # https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb
    idx = candles.index
    line_i = len(candles) - len(line_points)
    assert(line_i >= 0)
    points = []
    for i in range(line_i, len(candles)):
        points.append((idx[i], line_points[i - line_i]))
    return points

s_seq = get_line_points(candles, support_line)
r_seq = get_line_points(candles, resist_line)
s_seq2 = get_line_points(candles, support_line_c)
r_seq2 = get_line_points(candles, resist_line_c)
mpf.plot(candles, alines=dict(alines=[s_seq, r_seq, s_seq2, r_seq2], colors=['w', 'w', 'b', 'b']), type='candle', style='charles', ax=ax)
plt.show()
'''



def analyse(filtered_df: pd.DataFrame, 
            symbol: str, bot: TradingBot, 
            account_balance: float,
            lot_size: float,
            timeframe,
            ):
    
    total_trades = 0
    unexecuted_trades = 0
    successful_trades = 0
    unsuccessful_trades = 0
    gross_profit = 0
    loss = 0
    executed_trades = [] 
    break_even = 0
    conversion = bot.timeframe_to_interval.get(timeframe, 3600)
    
    
    for index, row in filtered_df.iterrows():
        #if trade is in valid, no further processing
        
        if check_invalid_stopouts(row):
            unexecuted_trades += 1
            print(f"trade invalid stopouts: {row['time']}")
            continue

        # allways add 15 min to start time because position was started at cnadle close and not open
        start_time= pd.to_datetime(row['time'] + pd.Timedelta(seconds=1)).ceil(conversion) #add 1 second to be able to apply ceil function
        end_time =  start_time + pd.Timedelta(days=4)
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
        row['time_updated'] = None
        if time_to_trail == pd.Timestamp.max and time_tp_hit == pd.Timestamp.max and time_sl_hit == pd.Timestamp.max:
            row['sl_updated'] = False
        else:
            row['sl_updated'] = min(time_sl_hit, time_tp_hit, time_to_trail) == time_to_trail
            
        

        #update actual sl and refind teh indexes
        if  row['sl_updated']:
            row['time_updated'] = time_to_trail
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

 
    executed_trades_df = pd.DataFrame(executed_trades)
    if loss != 0:
        profit_factor = gross_profit / abs(loss)
    else:
        profit_factor = float('inf')  # Handle case where there are no losing trades

    if total_trades > 0:
        percentage_profitability = ((successful_trades+break_even) / (total_trades)) * 100
    else:
        percentage_profitability = 0  # Handle case where there are no trades
        
    weekly_profit, monthly_profit = aggregate_profit(executed_trades_df=executed_trades_df)
    return {
        "percentage_profitability": percentage_profitability,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "loss": loss,        
        "total_trades": total_trades,
        "unexecuted_trades": unexecuted_trades,
        "unsuccessful_trades": unsuccessful_trades,
        "successful_trades": successful_trades,
        "break_even": break_even,
        "executed_trades_df": executed_trades_df,
        "weekly_profit": weekly_profit,
        "monthly_profit": monthly_profit,        
    }

def aggregate_profit(executed_trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    return weekly_df, monthly_df
