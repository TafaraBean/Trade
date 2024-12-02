import pandas as pd
import pandas_ta as ta
import numpy as np
import MetaTrader5 as mt5
import talib
from main import bot
from analysis import auto_trendline_15, auto_trendline, auto_trendline_4H
from concurrent.futures import ThreadPoolExecutor

def h1_gold_strategy(data):
    data['ema_short'] = ta.ema(data['close'], length=12)
    data['ema_long'] = ta.ema(data['close'], length=26)
    data['lsma'] = ta.linreg(data['close'], length=100)
    macd = ta.macd(data['close'], fast=15, slow=20, signal=4)
    data['macd_line'] = macd['MACD_15_20_4']
    data['lsma_stddev'] = data['close'].rolling(window=25).std()
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.6)
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.6)

    data['is_buy2'] = False
    data['is_sell2'] = True



    data.loc[data['is_buy2'], 'tp'] = data['close'] + 9
    data.loc[data['is_buy2'], 'sl'] = data['low'] - 3
    data.loc[data['is_sell2'], 'tp'] = data['close'] - 9
    data.loc[data['is_sell2'], 'sl'] = data['high'] + 3

    #set trailling stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 3
    data.loc[data['is_sell2'], 'be'] = data['close'] - 3

    #condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + 4
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - 4
    return data



def apply_strategy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = bot.copy_chart_range(symbol=bot.symbol, timeframe=bot.timeframe, start=start, end=end)
    data=auto_trendline_15(data)

    #hour_data = bot.copy_chart_range(symbol=bot.symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)

    #hour_data=auto_trendline(hour_data)

    #hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline','prev_fixed_support_trendline','prev_fixed_resistance_trendline','prev_fixed_resistance_gradient','prev_fixed_support_gradient','ema_24','prev_stochk','prev_stochd','prev_hour_macd_signal','prev_psar','prev_psar_direction','prev_nadaraya_watson','prev_nadaraya_watson_trend','nadaraya_upper_envelope','nadaraya_lower_envelope','wma_50','prev_supertrend_dir','HSpan_A','HSpan_B','nadaraya_watson','prev_nadaraya_lower_band','prev_nadaraya_upper_band','prev_HSpan_A','prev_HSpan_B','support_trendline','resistance_trendline','support','resistance','is_buy','is_sell']]

    #hour_data.to_csv("csv/hour_data.csv",index=False)

    #data['hourly_time']=data['time'].dt.floor('h')

    #merged_data = pd.merge(data,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))

    # four_hour_data = bot.copy_chart_range(symbol=bot.symbol, timeframe=mt5.TIMEFRAME_H4, start=start, end=end)
    # four_hour_data = auto_trendline_4H(four_hour_data)
    # H4_data = four_hour_data[['time4h','prev_4HSpan_A','prev_4HSpan_B','prev_4H_sr_levels']]

    # data['4_hour_time']=data['time'].dt.floor('4h')


    # merged_data2 = pd.merge(data,H4_data, left_on='4_hour_time', right_on='time4h', suffixes=('', '_4h'))

    return m15_gold_strategy(data)


def m15_gold_strategy(data: pd.DataFrame) -> pd.DataFrame:
    data['ema_short'] = ta.ema(data['close'], length=12)
    data['ema_long'] = ta.ema(data['close'], length=26)
    
    
    macd = ta.macd(data['close'], fast=12, slow=24, signal=9)
    data['macd_line'] = macd['MACD_12_24_9']
    data['macd_signal'] = macd['MACDs_12_24_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=8).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    data['prev_lsma_slope_1'] = data['lsma_slope'].shift(1)
    data['prev_lsma_slope_2'] = data['lsma_slope'].shift(2)
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + 2*data['lsma_stddev']
    data['lsma_lower_band'] = data['lsma'] - 2*data['lsma_stddev']

    # Calculate stochastic oscillator
    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['stoch_k'] = stochastic['STOCHk_14_3_3']
    data['stoch_d'] = stochastic['STOCHd_14_3_3']

    

    

    
    pip_size = 1

    # Set TP and SL in terms of pips
    tp_pips = 2 * pip_size
    sl_pips = 1 * pip_size
    be_pips =  30 * pip_size
    data["be_increment"] = 2.0
    data["be_condition_increment"] = 3.0
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

        (data['close'].shift(1)>data['fixed_support_trendline_15'].shift(1))&
        (data['close'].shift(2)<data['fixed_support_trendline_15'].shift(1))&
        (data['ema_50'].shift(1)>data['fixed_support_trendline_15'].shift(1))

      
    )


    data['is_sell2'] = (
        
        (data['close'].shift(1)<data['fixed_resistance_trendline_15'].shift(1))&
        (data['close'].shift(2)>data['fixed_resistance_trendline_15'].shift(1))&
        (data['ema_50'].shift(1)<data['fixed_resistance_trendline_15'].shift(1))
        
        )
        
    


    

    
    data.loc[data['is_buy2'], 'signal'] = mt5.ORDER_TYPE_BUY
    data.loc[data['is_sell2'], 'signal'] = mt5.ORDER_TYPE_SELL 
    data.loc[data['is_buy2'], 'tp'] = data['close'] + tp_pips
    data.loc[data['is_buy2'], 'sl'] = data['close'] - sl_pips
    data.loc[data['is_sell2'], 'tp'] = data['close'] - tp_pips
    data.loc[data['is_sell2'], 'sl'] = data['close'] + sl_pips

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 2 *  pip_size 
    data.loc[data['is_sell2'], 'be'] = data['close'] - 2 * pip_size

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
    # Concatenate 'be_condition_increment', 'be_increment', 'be_condition' for rows where 'is_buy2' is True
    data.loc[data['is_buy2'], "conditions_arr"] = data[data['is_buy2']].apply(
        lambda row: f"{row['be_condition_increment']},{row['be_increment']},{row['be_condition']}", axis=1
    )

    # Concatenate 'be_condition_increment', 'be_increment', 'be_condition' for rows where 'is_sell2' is True
    data.loc[data['is_sell2'], "conditions_arr"] = data[data['is_sell2']].apply(
        lambda row: f"{row['be_condition_increment']},{row['be_increment']},{row['be_condition']}", axis=1
)
    return data


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


def check_retest(current_close, sr_level, previous_close, breakout_direction, tolerance=0.1):
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