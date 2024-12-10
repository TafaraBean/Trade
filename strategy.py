import pandas as pd
import pandas_ta as ta
import numpy as np
import MetaTrader5 as mt5
import talib
from analysis import auto_trendline_15
from concurrent.futures import ThreadPoolExecutor




def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    print("Analysing market")
    data=auto_trendline_15(df)
    return m15_gold_strategy(data)


def m15_gold_strategy(data: pd.DataFrame) -> pd.DataFrame:
    
    

    
    
    
    pip_size = 1

    # Set TP and SL in terms of pips
    tp_pips = 6 * pip_size
    sl_pips = 20* pip_size
    be_pips =   10 * pip_size
    data["be_increment"] = 3.0
    data["be_condition_increment"] = 5.0
    data['ticket'] = np.nan
    


    session_times = {
        'New York': (1, 10),  # 08:00 to 17:00 EST (13:00 to 22:00 UTC)
}
    data['sr_cross_signal_sell'] = data.apply(
    lambda row: check_sr_crossings_sell(
        row['close'],
        data['close'].shift(1).loc[row.name],  # Pass previous close value
        row['sr_levels']
    ),
    axis=1
)
    
    data['sr_cross_signal_buy'] = data.apply(
    lambda row: check_sr_crossings_buy(
        row['close'],
        data['close'].shift(1).loc[row.name],  # Pass previous close value
        row['sr_levels']
    ),
    axis=1
)

    data['in_session'] = data['time'].apply(lambda row_time: is_within_trading_hours(row_time, session_times))
    data['retest_buy'] = data.apply(
        lambda row: check_retest(
            row['close'],
            row['sr_levels'][0] if row['sr_levels'] else None,  # Use the first level for simplicity
            data['close'].shift(1).loc[row.name],
            'buy'
        ) if row['sr_cross_signal_buy'] else False,
        axis=1
    )

    data['retest_sell'] = data.apply(
        lambda row: check_retest(
            row['close'],
            row['sr_levels'][0] if row['sr_levels'] else None,
            data['close'].shift(1).loc[row.name],
            'sell'
        ) if row['sr_cross_signal_sell'] else False,
        axis=1
    )

    # Generate signals
    data['is_buy2'] = (


        (data['regression_channel_slope4'].shift(1)>0)&(data['regression_channel_slope4'].shift(2)<0)


    #    (data['close'].shift(1)>data['bb2_lower'].shift(1))&
    #    (data['low'].shift(2)<data['bb2_lower'].shift(2))&
    #    (data['bb2_lower'].shift(1)<data['bb_lower'].shift(1))&
    #    (data['ADX']<27)

    #     (data['close'].shift(1)>data['fixed_resistance_trendline_15'].shift(1))&
    #     (data['close'].shift(2)<data['fixed_resistance_trendline_15'].shift(2))&
    #     (data['ADX']>25)

    #     (data['close'].shift(1)>data['bb2_lower'].shift(1))&
    #     (data['close'].shift(2)<data['bb2_lower'].shift(2))&
    #     (data['+DI'].shift(1)>data['-DI'].shift(1))

    #     (data['+DI'].shift(1)>data['-DI'].shift(1))&(data['+DI'].shift(2)<data['-DI'].shift(2))

      
    #     (data['support_gradient']>0)&(data['resistance_gradient']>0)&(data['sr_cross_signal_buy'])
    )


    data['is_sell2'] = (

        (data['regression_channel_slope4'].shift(1)<0)&(data['regression_channel_slope4'].shift(2)>0)
        
    #    (data['close'].shift(1)<data['bb2_upper'].shift(1))&
    #    (data['high'].shift(2)>data['bb2_upper'].shift(2))&
    #    (data['bb2_upper'].shift(1)>data['bb_upper'].shift(1))&
    #    (data['ADX']<27)

    #     (data['close'].shift(1)<data['fixed_support_trendline_15'].shift(1))&
    #     (data['close'].shift(2)>data['fixed_support_trendline_15'].shift(2))&
    #     (data['ADX']>25)

    #     (data['close'].shift(1)<data['bb2_upper'].shift(1))&
    #     (data['close'].shift(2)>data['bb2_upper'].shift(2))&
    #     (data['+DI'].shift(1)<data['-DI'].shift(1))

    #     (data['+DI'].shift(1)<data['-DI'].shift(1))&(data['+DI'].shift(2)>data['-DI'].shift(2))


    #     (data['support_gradient']<0)&(data['resistance_gradient']<0)&(data['sr_cross_signal_sell'])
        
        )
        
    


    # data['is_sell2'] = False
    # data['is_buy2'] = True

    data.loc[data['is_buy2'], 'signal'] = mt5.ORDER_TYPE_BUY
    data.loc[data['is_sell2'], 'signal'] = mt5.ORDER_TYPE_SELL 
    data.loc[data['is_buy2'], 'tp'] = data['close'] + tp_pips
    data.loc[data['is_buy2'], 'sl'] = data['close'] - sl_pips
    data.loc[data['is_sell2'], 'tp'] = data['close'] - tp_pips
    data.loc[data['is_sell2'], 'sl'] = data['close'] + sl_pips

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 4 *  pip_size 
    data.loc[data['is_sell2'], 'be'] = data['close'] - 4 * pip_size

    # Condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + be_pips
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - be_pips

    
    #set the array of conditions
    """
    in the conditions array, the following is stored at these indexes
    index 0: be_condition_increment
    index 1: be_increment
    index 2: be_condition

    """
        # Set the 'conditions_arr' for rows where 'is_buy2' is True
    data.loc[data['is_buy2'], "conditions_arr"] = data[data['is_buy2']].apply(concatenate_conditions, axis=1)

    # Set the 'conditions_arr' for rows where 'is_sell2' is True
    data.loc[data['is_sell2'], "conditions_arr"] = data[data['is_sell2']].apply(concatenate_conditions, axis=1)

    return data


def concatenate_conditions(row):
    return f"{row['be_condition_increment']},{row['be_increment']},{row['be_condition']}"

def is_within_trading_hours(row_time, session_times):
    hour = row_time.hour
    for session, (start, end) in session_times.items():
        if start <= hour <= end:
            return True
    return False


def check_sr_crossings_sell(current_close, previous_close, sr_levels):
    # Ensure sr_levels is iterable
    if not sr_levels:
        return None  # No signal if sr_levels is None or empty
    
    for level in sr_levels:
        # Check if the price has crossed below the level
        if current_close < level and previous_close >= level:
            return 'sell'  # Trigger a sell signal
    return None  # No signal


def check_sr_crossings_buy(current_close, previous_close, sr_levels):
    # Ensure sr_levels is iterable
    if not sr_levels:
        return None  # No signal if sr_levels is None or empty
    
    for level in sr_levels:
        # Check if the price has crossed above the level
        if current_close > level and previous_close <= level:
            return 'buy'  # Trigger a buy signal
    return None  # No signal


def check_retest(current_close, sr_level, previous_close, breakout_direction, tolerance=0.8):
    """
    Check if the price is retesting a support or resistance level.

    Args:
        current_close: The current close price.
        sr_level: The support or resistance level being tested.
        previous_close: The previous close price.
        breakout_direction: 'buy' for support retest, 'sell' for resistance retest.
        tolerance: The allowable range near the level to consider as a retest.

    Returns:
        True if retest condition is met, False otherwise.
    """
    if breakout_direction == 'buy':
        # Retest of support
        return sr_level - tolerance <= current_close <= sr_level + tolerance and current_close > previous_close
    elif breakout_direction == 'sell':
        # Retest of resistance
        return sr_level - tolerance <= current_close <= sr_level + tolerance and current_close < previous_close
    return False