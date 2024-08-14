import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm
from typing import Tuple
import calendar


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
    if invalid_stopouts:
        if is_buy2:
            print(f"Details:\nType: buy \nsl invalid: {(stop_loss >= close_price) }\ntp invalid: {(take_profit <= close_price)}")
        else:
            print(f"Details:\nType: sell \nsl invalid: {(stop_loss <= close_price) }\ntp invald: {(take_profit >= close_price)}")
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
                            y=df['nadaraya_upper_envelope'], 
                            mode='lines', 
                            name='nadaraya Up'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['lsma_upper_band'], 
                            mode='lines', 
                            name='Lsma_up'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['lsma_lower_band'], 
                            mode='lines', 
                            name='lsma Low'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['ema_50'], 
                            mode='lines', 
                            name='ema_50'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['wma_50'], 
                            mode='lines', 
                            name='wma_50'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['ema_24'], 
                            mode='lines', 
                            name='ema_24'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['fixed_support_trendline'], 
                            mode='lines', 
                            name='support'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'], 
                            y=df['fixed_resistance_trendline'], 
                            mode='lines', 
                            name='resistance'
                            ), row=1, col=1)

    # Add LMSA Lower Band line to the first subplot
    fig.add_trace(go.Scatter(x=df['time'],
                            y=df['nadaraya_lower_envelope'], 
                            mode='lines', 
                            name='nadaraya low'
                            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['time'],
                            y=df['nadaraya_watson'], 
                            mode='lines', 
                            name='nadaraya'
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
            symbol: str, 
            bot, 
            account_balance: float,
            lot_size: float,
            timeframe,
            ):
    
    total_trades = 0
    unexecuted_trades = 0
    successful_trades = 0
    unsuccessful_trades = 0
    running_trades = 0
    gross_profit = 0
    loss = 0
    executed_trades = [] 
    break_even = 0
    conversion = bot.timeframe_to_interval.get(timeframe, 3600)
    

    
    for index, row in filtered_df.iterrows():
        #if trade is in valid, no further processing
        
        print(f"Currently Working on Trade: {row['time']}", end=" ")
        if check_invalid_stopouts(row):
            unexecuted_trades += 1
            print(f"trade invalid stopouts: {row['time']}")
            continue

        #check for mixed signals
        if row['is_buy2'] and row['is_sell2']:
            print("Trade was both buy and sell")
            unexecuted_trades+=1
            continue

        # allways add 15 min to start time because position was started at cnadle close and not open
        start_time= pd.to_datetime(row['time'] + pd.Timedelta(seconds=1))
        start_time = start_time.ceil(conversion) #add 1 second to be able to apply ceil function

        end_time = start_time + pd.Timedelta(days=8)

        
        row['order_type'] = mt5.ORDER_TYPE_BUY if row['is_buy2'] else mt5.ORDER_TYPE_SELL
        row['lot'] = lot_size
        
        second_chart = pd.DataFrame(mt5.copy_rates_range(symbol, timeframe, start_time, end_time))
        relevant_ticks = pd.DataFrame(mt5.copy_ticks_range(symbol, start_time, end_time, mt5.COPY_TICKS_ALL))
        if len(second_chart) != 0:
            second_chart['time'] = pd.to_datetime(second_chart['time'],unit='s')
        if len(relevant_ticks) != 0:
            relevant_ticks['time'] = pd.to_datetime(relevant_ticks['time'],unit='s')

        static_ticks = relevant_ticks# for later use when check if trade was manually closed
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
            
        
        sl_updated = 0
        timestamps_list = []# to keep track of all the times stop losses were updated
        #update actual sl and refind teh indexes
        if  row['sl_updated']:
            while (min(time_sl_hit, time_to_trail, time_tp_hit)== time_to_trail and not(time_sl_hit== pd.Timestamp.max and time_tp_hit == pd.Timestamp.max and time_to_trail == pd.Timestamp.max )):
                sl_updated+=1
                relevant_ticks = relevant_ticks[relevant_ticks['time'] >= time_to_trail]
                second_chart = second_chart[second_chart['time'] >= time_to_trail]
                
                relevant_ticks.reset_index(drop=True, inplace=True)
                second_chart.reset_index(drop=True, inplace=True)

                #update the stop loss level
                row['sl'] = row['be']

                if row['is_buy2']:
                    row['be'] += 20 * 0.0001 
                    row['be_condition'] += 30 * 0.0001
                else:
                    row['be'] -= 20 * 0.0001 
                    row['be_condition'] -= 30* 0.0001

                stop_loss_reached = relevant_ticks['bid'] <= row['sl'] if row['is_buy2'] else relevant_ticks['bid'] >= row['sl']
                trailing_stop_reached = (second_chart['close'] >= row["be_condition"]) if row["is_buy2"] else (second_chart['close'] <= row["be_condition"])

                #find the new time for ehich stop loss was hit
                stop_loss_index = np.argmax(stop_loss_reached) if stop_loss_reached.any() else -1
                trailing_stop_index = np.argmax(trailing_stop_reached) if trailing_stop_reached.any() else -1
                #add timestamp before updating it
                timestamps_list.append(time_to_trail)
                time_sl_hit = relevant_ticks.loc[stop_loss_index, 'time'] if stop_loss_index != -1 else pd.Timestamp.max
                time_to_trail = (second_chart.loc[trailing_stop_index, "time"] + pd.Timedelta(seconds=1)).ceil(conversion) if trailing_stop_index != -1 else pd.Timestamp.max
                
        print(f"sl updated {sl_updated} times at")
        for timestamp in timestamps_list:
            print(timestamp)
        
        #save final updated times
        row['time_to_trail'] = pd.NA if time_to_trail == pd.Timestamp.max else time_to_trail
        row['time_tp_hit'] = pd.NA if time_tp_hit == pd.Timestamp.max else time_tp_hit
        row['time_sl_hit'] = pd.NA if time_sl_hit == pd.Timestamp.max else time_sl_hit

        row['entry_time'] = (row['time'] + pd.Timedelta(seconds=1)).ceil(conversion)
        row['entry_price'] = row['close']

        row['exit_time'] = min(time_sl_hit, time_tp_hit) 
        


        
        matching_row = relevant_ticks[relevant_ticks['time'] == row['exit_time']]
        row['exit_time'] = pd.NA if row['exit_time'] == pd.Timestamp.max else row['exit_time']
        if row['exit_time'] == pd.NA or not matching_row.empty:
            row['exit_price'] = matching_row.iloc[0]['bid']
        else:
            row['exit_price'] =  pd.NA

        
        following_signals = filtered_df[
            ((filtered_df['time'] >= row['entry_time']) & 
            (filtered_df['time'] <= row['exit_time']) & 
            (filtered_df['is_buy2'] != row['is_buy2']))
        ]

        if not following_signals.empty:
            first_signal_time = (following_signals.iloc[0]['time'] + pd.Timedelta(seconds=1)).ceil(conversion)
            print(f"following signal: {first_signal_time}")
            
            if min(first_signal_time,time_sl_hit, time_tp_hit) == first_signal_time:
                
                # Try to find the index of the first row that matches the first_signal_time
                matching_rows = static_ticks[static_ticks['time'] == first_signal_time]

                if not matching_rows.empty:
                    # If a match is found, use its index
                    start_index = matching_rows.index[0]
                else:
                    # If no match is found, use the next available row
                    start_index = static_ticks[static_ticks['time'] > first_signal_time].index[0]


                #row['exit_price'] = static_ticks.iloc[start_index]['bid']
                temp_exit_price= static_ticks.iloc[start_index]['bid']
                temp_profit =  bot.profit_loss(symbol=symbol, order_type=row['order_type'], lot=lot_size, open_price=row["entry_price"], close_price=temp_exit_price) 

                if temp_profit>0:
                    row['exit']= "auto"
                else:
                    row['exit_price'] = static_ticks.iloc[start_index]['bid']
                    row['exit_time'] = first_signal_time
                    row['exit'] ="manual" 
            else:
                row['exit']= "auto"

        else:
            print(f"following signal: {None}")
            first_signal_time = pd.NA
            row['exit'] ="auto" 

        #set exit time  to none after use
        
        print(f"type: {row['exit']}")
        print(f"ex time: {row['exit_time']}")
        print(f"tp time: {row['time_tp_hit']}")
        print(f"sl time: {row['time_sl_hit'] }")
        print(f"tr time: {row['time_to_trail']}")
        
        

        if stop_loss_index == 0 or take_profit_index == 0:
            print(f"take profit or stop loss reached ar zero for trade {row['time']}")
            unexecuted_trades +=1
            continue
        
        total_trades+=1
        executed_trades.append(row)
        row['lot_size'] = lot_size
        #set the value for the type of trade this was, weather loss, even or success
        if row['exit'] == "manual":
            pass
        elif stop_loss_index > -1 and take_profit_index > -1:
            if(min(time_sl_hit, time_tp_hit) == time_tp_hit):                
                row['type'] = "success"
                successful_trades+=1
          
            elif(row['sl_updated']):       
                row['type'] = "even"
                break_even +=1

            else:
                row['type'] = "fail"
                unsuccessful_trades +=1            

        elif take_profit_index == -1 and stop_loss_index != -1:
            if(row['sl_updated']):
                row['type'] = "even"
                break_even +=1

            else:
                row['type'] = "fail"
                unsuccessful_trades+=1            

        elif stop_loss_index == -1 and take_profit_index != -1:
            successful_trades+=1
            row['type'] = "success"

        else:
            row['type'] = "running"
            running_trades += 1

        
        
      #calculate its profit value
        if row['exit_time'] != pd.NA:
            row['profit'] =  bot.profit_loss(symbol=symbol, order_type=row['order_type'], lot=lot_size, open_price=row["entry_price"], close_price=row["exit_price"]) 
        else:
            row['profit'] = 0

        if row['exit'] == "manual":
            row['type'] = 'success' if row['profit'] >= 0 else 'fail'
            if row['type'] == "success":
                successful_trades +=1
            else:
                unsuccessful_trades+=1
        
        account_balance  += row['profit']
        row["account_balance"] = account_balance
        
        if row['profit'] >= 0:
            gross_profit += row['profit']
        else:
            loss += row["profit"]
        
        row['success'] = row['type'] in['success', 'even']

        print(f"trade {row['type']}")

    
    executed_trades_df = pd.DataFrame(executed_trades)
    profit_factor = calc_profit_factor(gross_profit=gross_profit,
                                        loss=loss)
    percentage_profitability = calc_percentage_profitability(successful_trades=successful_trades,
                                                                break_even=break_even,
                                                                total_trades=total_trades)
    weekly_profit, monthly_profit = aggregate_profit(executed_trades_df=executed_trades_df)

 

    return {
        "account_balance": account_balance,
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
        "running_trades": running_trades,      
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
    # Convert month number to month name
    monthly_df['Month'] = monthly_df['Month'].apply(lambda x: calendar.month_name[x])
    return weekly_df, monthly_df

def calc_profit_factor(loss: float,
                       gross_profit: float):
    if loss != 0:
        profit_factor = gross_profit / abs(loss)
    else:
        profit_factor = float('inf')  # Handle case where there are no losing trades
    
    return profit_factor

def calc_percentage_profitability(successful_trades: int,
                       break_even: int,
                       total_trades: int):
    if total_trades > 0:
        percentage_profitability = ((successful_trades+break_even) / (total_trades)) * 100
    else:
        percentage_profitability = 0  # Handle case where there are no trades
    
    return percentage_profitability


def auto_trendline_15(data: pd.DataFrame) -> pd.DataFrame:
    print("applying auto trendline...")
    data['time1'] = data['time'].astype('datetime64[s]')
    data = data.set_index('time1', drop=True)
    print("15-min data:")
    #print(data)
    print("==========")

    # Take natural log of data to resolve price scaling issues
    df_log = np.log(data[['high', 'low', 'close']])

    # Trendline parameter
    lookback = 20

    # Initialize columns for trendlines and their gradients
    data['support_trendline_15'] = np.nan
    data['resistance_trendline_15'] = np.nan
    data['support_gradient_15'] = np.nan
    data['resistance_gradient_15'] = np.nan
    data['supertrend_dir']=np.nan
    data['Span_A']=np.nan
    data['Span_B']=np.nan


    lookback2 = 50
    for i in range(lookback2, len(df_log)+1):
        current_index = df_log.index[i-1]
        window_data = df_log.iloc[:i]
        supertrend_result = ta.supertrend(window_data['high'],window_data['low'], window_data['close'], length=20, multiplier=3)
        data.at[current_index,'supertrend_dir']=supertrend_result['SUPERTd_20_3.0'].iloc[-1]
        ichi = ta.ichimoku(window_data['high'],window_data['low'],window_data['close'])
        look_ahead_spans = ichi[1]
        
        if look_ahead_spans is not None:
            senkou_span_a = look_ahead_spans['ISA_9']
            senkou_span_b = look_ahead_spans['ISB_26']
            data.at[current_index, 'Span_A'] = senkou_span_a.iloc[-1]
            data.at[current_index, 'Span_B'] = senkou_span_b.iloc[-1]
        else:
            data.at[current_index, 'Span_A'] = None
            data.at[current_index, 'Span_B'] = None

        
        # Determine trend based on Nadaraya-Watson
        
        
    #ISA_9    ISB_26



    # Iterate over the dataset in overlapping windows of 15 candles
    for i in range(lookback, len(df_log) + 1):
        current_index = df_log.index[i-1]
        window_data = df_log.iloc[i - lookback:i]
        support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
        

        
        
        # Extract slope and intercept
        support_slope, support_intercept = support_coefs
        resist_slope, resist_intercept = resist_coefs
        data.at[current_index, 'fixed_resistance_gradient_15'] = resist_slope
        data.at[current_index, 'fixed_support_gradient_15'] = support_slope
        support_value = support_slope * window_data.at[current_index,'close'] + support_intercept
        resist_value = resist_slope * window_data.at[current_index,'close'] + resist_intercept
        data.at[current_index, 'fixed_support_trendline_15'] = np.exp(support_value)
        data.at[current_index, 'fixed_resistance_trendline_15'] = np.exp(resist_value)
        # Apply the calculated gradients to each candle in the window
        
        
        for j in range(lookback):
            idx = i - lookback + j
            support_value = support_slope * j + support_intercept
            resist_value = resist_slope * j + resist_intercept
            data.at[data.index[idx], 'support_trendline_15'] = np.exp(support_value)
            data.at[data.index[idx], 'resistance_trendline_15'] = np.exp(resist_value)
            data.at[data.index[idx], 'support_gradient_15'] = support_slope
            data.at[data.index[idx], 'resistance_gradient_15'] = resist_slope
    return data

def nadaraya_watson_smoother(x, y, bandwidth):
    """Nadaraya-Watson kernel regression smoother."""
    n = len(x)
    y_hat = np.zeros(n)
    
    for i in range(n):
        weights = norm.pdf((x - x[i]) / bandwidth)
        weights /= weights.sum()
        y_hat[i] = np.sum(weights * y)
    
    return y_hat

def determine_trend(current_value, previous_value):
    """Determine market trend based on the current and previous smoothed prices."""
    if np.isnan(previous_value):
        return np.nan  # Not enough data to determine trend
    return 'bullish' if current_value > previous_value else 'bearish'


def auto_trendline(data: pd.DataFrame) -> pd.DataFrame:
    print("applying auto trendline...")
    data['time2'] = data['time'].astype('datetime64[s]')
    data = data.set_index('time', drop=True)
    print("hourly data:")
    print("==========")

    # Take natural log of data to resolve price scaling issues
    df_log = np.log(data[['high', 'low', 'close']])

    # Trendline parameter
    lookback = 60

    # Initialize columns for trendlines and their gradients
    data['support_trendline'] = np.nan
    data['resistance_trendline'] = np.nan
    data['support_gradient'] = np.nan
    data['resistance_gradient'] = np.nan

    data['ema_50'] = ta.ema(data['close'], length=80)
    data['ema_24'] = ta.ema(data['close'], length=8)
    data['hour_lsma'] = ta.linreg(data['close'], length=10)
    data['prev_hour_lsma'] = data['hour_lsma'].shift(1)
    data['hour_lsma_slope'] = data['hour_lsma'].diff()
    data['prev_hour_lsma_slope'] = data['hour_lsma_slope'].shift(1)
    data['wma_50'] = ta.wma(data['close'], length=100)

    macd = ta.macd(data['close'], fast=8, slow=17, signal=9)
    data['hour_macd_line'] = macd['MACD_8_17_9']
    data['hour_macd_signal'] = macd['MACDs_8_17_9']

    data['hstoch_k'] = np.nan
    data['hstoch_d'] = np.nan

    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['hstoch_k'] = stochastic['STOCHk_14_3_3']
    data['hstoch_d'] = stochastic['STOCHd_14_3_3']

    data['prev_stochk'] = data['hstoch_k'].shift(1)
    data['prev_stochd'] = data['hstoch_d'].shift(1)
    data['HSpan_A'] = np.nan
    data['HSpan_B'] = np.nan



    data['nadaraya_watson'] = np.nan
    data['nadaraya_watson_trend'] = np.nan
    data['nadaraya_upper_envelope']=np.nan
    data['nadaraya_lower_envelope']=np.nan
    data['supertrend'] = np.nan
    data['supertrend_dir'] = np.nan
    data['psar']=np.nan
    data['psar_direction']=np.nan

    bandwidth = 10  # Adjust as needed

    previous_value = np.nan
    
    lookback2=60
    
    for i in range(lookback2, len(df_log) + 1):
        current_index = df_log.index[i - 1]
        window_data = df_log.iloc[i-lookback:i]  # Use all data up to the current point
        supertrend_result = ta.supertrend(window_data['high'],window_data['low'], window_data['close'], length=2, multiplier=3)
        ichi = ta.ichimoku(window_data['high'],window_data['low'],window_data['close'])
        look_ahead_spans = ichi[1]
        
        if look_ahead_spans is not None:
            senkou_span_a = look_ahead_spans['ISA_9']
            senkou_span_b = look_ahead_spans['ISB_26']
            data.at[current_index, 'HSpan_A'] = senkou_span_a.iloc[-1]
            data.at[current_index, 'HSpan_B'] = senkou_span_b.iloc[-1]
        else:
            data.at[current_index, 'HSpan_A'] = None
            data.at[current_index, 'HSpan_B'] = None

        data.at[current_index,'supertrend'] = supertrend_result['SUPERT_2_3.0'].iloc[-1]
        data.at[current_index,'supertrend_dir']=supertrend_result['SUPERTd_2_3.0'].iloc[-1]

        
        x = np.arange(len(window_data))
        y = window_data['close'].values
        y_smoothed_log = nadaraya_watson_smoother(x, y, bandwidth)
        current_value_log = y_smoothed_log[-1]
        
        # Convert back to original scale
        y_smoothed = np.exp(y_smoothed_log)
        current_value = np.exp(current_value_log)
        
        # Calculate residuals
        residuals = window_data['close'] - y_smoothed_log
        std_residuals = np.std(residuals)
        
        # Compute upper and lower envelopes on the log scale
        upper_envelope_log = y_smoothed_log + 0.8 * std_residuals  # 3 standard deviations for example
        lower_envelope_log = y_smoothed_log - 0.8 * std_residuals

        # Convert envelopes back to original scale
        upper_envelope = np.exp(upper_envelope_log)
        lower_envelope = np.exp(lower_envelope_log)

        data.at[current_index, 'nadaraya_watson'] = current_value
        data.at[current_index, 'nadaraya_upper_envelope'] = upper_envelope[-1]
        data.at[current_index, 'nadaraya_lower_envelope'] = lower_envelope[-1]
        
        # Determine trend based on Nadaraya-Watson
        data.at[current_index, 'nadaraya_watson_trend'] = determine_trend(current_value, previous_value)
        previous_value = current_value
        
        psar_series = ta.psar(np.exp(window_data['high']), np.exp(window_data['low']), np.exp(window_data['close']), af=0.02, max_af=0.2)['PSARl_0.02_0.2']
        data.at[current_index, 'psar'] = psar_series.iloc[-1]
        
        # Determine PSAR trend direction
        if window_data['close'].iloc[-1] > psar_series.iloc[-1]:
            psar_direction = 1  # Bullish
        else:
            psar_direction = -1  # Bearish

        data.at[current_index, 'psar_direction'] = psar_direction
        
        

    for i in range(lookback, len(df_log) + 1):
        current_index = df_log.index[i - 1]
        window_data = df_log.iloc[i - lookback:i]

        # Calculate Nadaraya-Watson estimator using only past data
        

        

        support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])

        # Extract slope and intercept
        support_slope, support_intercept = support_coefs
        resist_slope, resist_intercept = resist_coefs
        data.at[current_index, 'fixed_resistance_gradient'] = resist_slope
        data.at[current_index, 'fixed_support_gradient'] = support_slope
        support_value = support_slope * window_data['low'].iloc[-1] + support_intercept
        resist_value = resist_slope * window_data['high'].iloc[-1] + resist_intercept
        data.at[current_index, 'fixed_support_trendline'] = np.exp(support_value)
        data.at[current_index, 'fixed_resistance_trendline'] = np.exp(resist_value)

        for j in range(lookback):
            idx = i - lookback + j
            support_value = support_slope * j + support_intercept
            resist_value = resist_slope * j + resist_intercept
            data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
            data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
            data.at[data.index[idx], 'support_gradient'] = support_slope
            data.at[data.index[idx], 'resistance_gradient'] = resist_slope

    data['prev_fixed_support_trendline'] = data['fixed_support_trendline'].shift(1)
    data['prev_fixed_resistance_trendline'] = data['fixed_resistance_trendline'].shift(1)
    data['prev_fixed_support_gradient'] = (data['fixed_support_gradient'].shift(1))
    data['prev_fixed_resistance_gradient'] = (data['fixed_resistance_gradient'].shift(1))
    data['prev_hour_macd_line'] = data['hour_macd_line'].shift(1)
    data['prev_hour_macd_signal'] = data['hour_macd_signal'].shift(1)
    data['prev_nadaraya_watson'] = data['nadaraya_watson'].shift(1)
    data['prev_nadaraya_lower_band'] = data['nadaraya_lower_envelope'].shift(1)
    data['prev_nadaraya_upper_band'] = data['nadaraya_upper_envelope'].shift(1)
    data['prev_nadaraya_watson_trend'] = data['nadaraya_watson_trend'].shift(1)
    data['prev_psar']=data['psar'].shift(1)
    data['prev_psar_direction']=data['psar_direction'].shift(1)
    data['prev_supertrend'] = data['supertrend'].shift(1)
    data['prev_supertrend_dir']=data['supertrend_dir'].shift(1)
    data['prev_HSpan_A'] = data['HSpan_A'].shift(1)
    data['prev_HSpan_B'] = data['HSpan_B'].shift(1)
    data.to_csv("csv/test.csv", index=False)
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


def auto_trendline_4H(data: pd.DataFrame) -> pd.DataFrame:
    print("applying auto trendline...")
    data['time4h'] = data['time'].astype('datetime64[s]')
    data = data.set_index('time', drop=True)
    print("4 hour data:")
    print("==========")

    # Take natural log of data to resolve price scaling issues
    df_log = np.log(data[['high', 'low', 'close']])

    # Trendline parameter
    lookback = 10

    # Initialize columns for trendlines and their gradients
    data['support_trendline'] = np.nan
    data['resistance_trendline'] = np.nan
    data['support_gradient'] = np.nan
    data['resistance_gradient'] = np.nan

    data['4H_ema_50'] = ta.ema(data['close'], length=5)
    data['4H_ema_24'] = ta.ema(data['close'], length=17)
    data['4H_hour_lsma'] = ta.linreg(data['close'], length=8)
    data['4lsma_stddev'] = data['close'].rolling(window=10).std()
    data['4lsma_upper_band'] = data['4H_hour_lsma'] + 1.5*data['4lsma_stddev']
    data['4lsma_lower_band'] = data['4H_hour_lsma'] - 1.5* data['4lsma_stddev']
    data['4H_prev_hour_lsma'] = data['4H_hour_lsma'].shift(1)
    data['4H_hour_lsma_slope'] = data['4H_hour_lsma'].diff()
    data['4H_prev_hour_lsma_slope'] = data['4H_hour_lsma_slope'].shift(1)
    data['4H_wma_10'] = ta.wma(data['close'], length=17)

    macd = ta.macd(data['close'], fast=8, slow=17, signal=9)
    data['4hour_macd_line'] = macd['MACD_8_17_9']
    data['4hour_macd_signal'] = macd['MACDs_8_17_9']

    data['4stoch_k'] = np.nan
    data['4stoch_d'] = np.nan

    data['4nadaraya_watson'] = np.nan
    data['4nadaraya_watson_trend'] = np.nan
    data['4nadaraya_upper_envelope']=np.nan
    data['4nadaraya_lower_envelope']=np.nan
    data['4psar']=np.nan
    data['4psar_direction']=np.nan
    data['4HH']=np.nan
    data['4LL']=np.nan


    bandwidth = 10  # Adjust as needed

    previous_value = np.nan
    
    lookback2=9
    
    for i in range(lookback2, len(df_log) + 1):
        current_index = df_log.index[i - 1]
        window_data = df_log.iloc[:i]  # Use all data up to the current point

        ichi = ta.ichimoku(window_data['high'],window_data['low'],window_data['close'])
        look_ahead_spans = ichi[1]
        
        if look_ahead_spans is not None:
            senkou_span_a = look_ahead_spans['ISA_9']
            senkou_span_b = look_ahead_spans['ISB_26']
            data.at[current_index, '4HSpan_A'] = senkou_span_a.iloc[-1]
            data.at[current_index, '4HSpan_B'] = senkou_span_b.iloc[-1]
        else:
            data.at[current_index, '4HSpan_A'] = None
            data.at[current_index, '4HSpan_B'] = None

        
        
        x = np.arange(len(window_data))
        y = window_data['close'].values
        y_smoothed_log = nadaraya_watson_smoother(x, y, bandwidth)
        current_value_log = y_smoothed_log[-1]
        
        # Convert back to original scale
        y_smoothed = np.exp(y_smoothed_log)
        current_value = np.exp(current_value_log)
        
        # Calculate residuals
        residuals = window_data['close'] - y_smoothed_log
        std_residuals = np.std(residuals)
        
        # Compute upper and lower envelopes on the log scale
        upper_envelope_log = y_smoothed_log + 2 * std_residuals  # 3 standard deviations for example
        lower_envelope_log = y_smoothed_log - 2 * std_residuals

        # Convert envelopes back to original scale
        upper_envelope = np.exp(upper_envelope_log)
        lower_envelope = np.exp(lower_envelope_log)

        data.at[current_index, '4nadaraya_watson'] = current_value
        data.at[current_index, '4nadaraya_upper_envelope'] = upper_envelope[-1]
        data.at[current_index, '4nadaraya_lower_envelope'] = lower_envelope[-1]
        
        # Determine trend based on Nadaraya-Watson
        data.at[current_index, '4nadaraya_watson_trend'] = determine_trend(current_value, previous_value)
        previous_value = current_value
        
        psar_series = ta.psar(np.exp(window_data['high']), np.exp(window_data['low']), np.exp(window_data['close']), af=0.02, max_af=0.2)['PSARl_0.02_0.2']
        data.at[current_index, '4psar'] = psar_series.iloc[-1]
        
        # Determine PSAR trend direction
        if window_data['close'].iloc[-1] > psar_series.iloc[-1]:
            psar_direction = 1  # Bullish
        else:
            psar_direction = -1  # Bearish

        data.at[current_index, '4psar_direction'] = psar_direction
        
        

    for i in range(lookback, len(df_log) + 1):
        current_index = df_log.index[i - 1]
        window_data = df_log.iloc[i - lookback:i]

        # Calculate Nadaraya-Watson estimator using only past data
        

        

        support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])

        # Extract slope and intercept
        support_slope, support_intercept = support_coefs
        resist_slope, resist_intercept = resist_coefs
        data.at[current_index, '4fixed_resistance_gradient'] = resist_slope
        data.at[current_index, '4fixed_support_gradient'] = support_slope
        support_value = support_slope * window_data['close'].iloc[-1] + support_intercept
        resist_value = resist_slope * window_data['close'].iloc[-1] + resist_intercept
        data.at[current_index, '4fixed_support_trendline'] = np.exp(support_value)
        data.at[current_index, '4fixed_resistance_trendline'] = np.exp(resist_value)

        for j in range(lookback):
            idx = i - lookback + j
            support_value = support_slope * j + support_intercept
            resist_value = resist_slope * j + resist_intercept
            data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
            data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
            data.at[data.index[idx], 'support_gradient'] = support_slope
            data.at[data.index[idx], 'resistance_gradient'] = resist_slope

    data['prev_4fixed_support_trendline'] = data['4fixed_support_trendline'].shift(1)
    data['prev_4fixed_resistance_trendline'] = data['4fixed_resistance_trendline'].shift(1)
    data['prev_4fixed_support_gradient'] = (data['4fixed_support_gradient'].shift(1))
    data['prev_4fixed_resistance_gradient'] = (data['4fixed_resistance_gradient'].shift(1))
    data['prev_4hour_macd_line'] = data['4hour_macd_line'].shift(1)
    data['prev_4hour_macd_signal'] = data['4hour_macd_signal'].shift(1)
    data['prev_4nadaraya_watson'] = data['4nadaraya_watson'].shift(1)
    data['prev_4nadaraya_watson_trend'] = data['4nadaraya_watson_trend'].shift(1)
    data['prev_4psar']=data['4psar'].shift(1)
    data['prev_4psar_direction']=data['4psar_direction'].shift(1)
    data.to_csv("csv/4test.csv", index=False)
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


# def profit_loss(lot_size, open_price, close_price, is_buy):
#     """
#     This function calculates the profit or loss for a given trade.

#     Args:
#         lot_size: The number of units bought or sold (e.g., shares, contracts).
#         open_price: The price at which the trade was opened.
#         close_price: The price at which the trade was closed.
#         is_buy: A boolean flag indicating whether the trade was a buy (True) or sell (False).

#     Returns:
#         The profit or loss for the trade. A positive value indicates profit, 
#         and a negative value indicates loss.
#     """

#     if is_buy:
#         # Profit for buy trade = (close price - open price) * lot size
#         profit_loss = (close_price - open_price) * lot_size
#     else:
#         # Profit for sell trade = (open price - close price) * lot size
#         profit_loss = (open_price - close_price) * lot_size

#     return profit_loss