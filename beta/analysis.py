import pandas as pd
import MetaTrader5 as mt5

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