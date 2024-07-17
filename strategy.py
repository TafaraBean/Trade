import pandas as pd
import pandas_ta as ta

def h1_gold_strategy(data):
        data['ema_short'] = ta.ema(data['close'], length=12)
        data['ema_long'] = ta.ema(data['close'], length=26)
        data['lsma'] = ta.linreg(data['close'], length=50)
        macd = ta.macd(data['close'], fast=15, slow=20, signal=4)
        data['macd_line'] = macd['MACD_15_20_4']
        data['lsma_stddev'] = data['close'].rolling(window=25).std()
        data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.6)
        data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.6)

        data['is_buy2'] = (data['low'] < data['lsma_lower_band']) & (data['open'] < data['close']) & \
                        (data['open'].shift(1) > data['close'].shift(1)) & (data['macd_line'] < 0)
        data['is_sell2'] = (data['high'] > data['lsma_upper_band']) & (data['open'] > data['close']) & \
                        (data['open'].shift(1) < data['close'].shift(1)) & (data['macd_line'] > 0)



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




def m15_gold_strategy(data: pd.DataFrame) -> pd.DataFrame:
    data['ema_short'] = ta.ema(data['close'], length=12)
    data['ema_long'] = ta.ema(data['close'], length=26)
    data['lsma'] = ta.linreg(data['close'], length=18)
    
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data['macd_line'] = macd['MACD_12_26_9']
    data['macd_signal'] = macd['MACDs_12_26_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=25).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    data['prev_lsma_slope_1'] = data['lsma_slope'].shift(1)
    data['prev_lsma_slope_2'] = data['lsma_slope'].shift(2)
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] ) 
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] ) 

    # Calculate stochastic oscillator
    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['stoch_k'] = stochastic['STOCHk_14_3_3']
    data['stoch_d'] = stochastic['STOCHd_14_3_3']



    pip_size = 0.0001

    # Set TP and SL in terms of pips
    tp_pips = 50 * pip_size  # e.g., 50 pips
    sl_pips = 90 * pip_size  # e.g., 20 pips
    be_pips = 10 * pip_size

    # Generate signals
    data['is_buy2'] = (
        ((data['fixed_support_trendline_15'] < data['prev_fixed_support_trendline'].shift(1)) &
        (data['open'] < data['close']) &
        (data['prev_psar_direction'] == 1)) | (
        (data['close'].shift(1) > data['prev_fixed_resistance_trendline'].shift(1)) & 
        (abs(data['open'] - data['close']) < 300) & 
        (data['ema_50'] < data['close']) & 
        (data['stoch_k'] > 80) & 
        (data['stoch_k'] > data['stoch_d']) & 
        (data['prev_hour_macd_line'] > data['prev_hour_macd_signal'])
    ) # Only buy if PSAR indicates an uptrend
    )

    data['is_sell2'] = (
        ((data['fixed_resistance_trendline_15'] > data['prev_fixed_resistance_trendline'].shift(1)) &
        (data['open'] > data['close']) &
        (data['prev_psar_direction'] == -1)) | (
        (data['close'].shift(1) < data['prev_fixed_support_trendline'].shift(1)) & 
        (abs(data['open'] - data['close']) < 300) & 
        (data['ema_50'] > data['close']) & 
        (data['stoch_k'] < 20) & 
        (data['stoch_k'] < data['stoch_d']) & 
        (data['prev_hour_macd_line'] < data['prev_hour_macd_signal'])
    )  # Only sell if PSAR indicates a downtrend
    )
    
    data.loc[data['is_buy2'], 'tp'] = data['close'] + tp_pips
    data.loc[data['is_buy2'], 'sl'] = data['close'] - sl_pips
    data.loc[data['is_sell2'], 'tp'] = data['close'] - tp_pips
    data.loc[data['is_sell2'], 'sl'] = data['close'] + sl_pips

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 8 * pip_size
    data.loc[data['is_sell2'], 'be'] = data['close'] - 8 * pip_size

    # Condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + be_pips
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - be_pips

    return data