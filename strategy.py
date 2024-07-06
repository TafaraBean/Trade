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
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35) + (data['lsma_slope'] >= 0) * 1.5
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35) - (data['lsma_slope'] <= 0) * 1.5

    # Calculate stochastic oscillator
    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['stoch_k'] = stochastic['STOCHk_14_3_3']
    data['stoch_d'] = stochastic['STOCHd_14_3_3']

    # Generate signals
    data['is_buy2'] = (
        (data['close'].shift(1) > data['prev_fixed_resistance_trendline'].shift(1)) & 
        (abs(data['open'] - data['close']) < 300) & 
        (data['ema_50'] < data['close']) & 
        (data['stoch_k'] > 80) & 
        (data['stoch_k'] > data['stoch_d']) & 
        (data['prev_hour_macd_line'] > data['prev_hour_macd_signal'])
    )

    data['is_sell2'] = (
        (data['close'].shift(1) < data['prev_fixed_support_trendline'].shift(1)) & 
        (abs(data['open'] - data['close']) < 300) & 
        (data['ema_50'] > data['close']) & 
        (data['stoch_k'] < 20) & 
        (data['stoch_k'] < data['stoch_d']) & 
        (data['prev_hour_macd_line'] < data['prev_hour_macd_signal'])
    )
    
    data.loc[data['is_buy2'], 'tp'] = data['fixed_resistance_trendline']
    data.loc[data['is_buy2'], 'sl'] = data['prev_fixed_support_trendline']
    data.loc[data['is_sell2'], 'tp'] = data['fixed_support_trendline']
    data.loc[data['is_sell2'], 'sl'] = data['prev_fixed_resistance_trendline']

    # Ensuring tp is valid for buy and sell orders
    data.loc[(data['is_buy2']) & (data['tp'] <= data['close']+250), 'tp'] = data['close'] + 500
    data.loc[(data['is_sell2']) & (data['tp'] >= data['close']-250), 'tp'] = data['close'] - 500    

    # Ensuring sl is valid for buy and sell orders
    data.loc[(data['is_buy2']) & (data['sl'] >= data['close'])-150, 'sl'] = data['close'] - 350
    data.loc[(data['is_sell2']) & (data['sl'] <= data['close']+150), 'sl'] = data['close'] + 350  

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 350
    data.loc[data['is_sell2'], 'be'] = data['close'] - 350

    # Condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + 400
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - 400

    # Adjust signals based on SL and TP distances
    #data['is_buy2'] = data['is_buy2'] & (
    #    abs(data['sl'] - data['close']) <= 2.8 * abs(data['tp'] - data['close'])
    #)
    #data['is_sell2'] = data['is_sell2'] & (
    #    abs(data['sl'] - data['close']) <= 2.8 * abs(data['tp'] - data['close'])
    #)

    return data