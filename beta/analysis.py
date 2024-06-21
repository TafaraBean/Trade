import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
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

