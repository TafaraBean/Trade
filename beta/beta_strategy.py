import pandas as pd
import pandas_ta as ta

def h1_gold_strategy(data):
        data['ema_short'] = ta.ema(data['close'], length=12)
        data['ema_long'] = ta.ema(data['close'], length=26)
        data['lsma'] = ta.linreg(data['close'], length=30)
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



def m15_gold_strategy(data):
    # Calculate indicators
    data['ema_short'] = ta.ema(data['close'], length=12)
    data['ema_long'] = ta.ema(data['close'], length=26)
    data['lsma'] = ta.linreg(data['close'], length=25)
    
    macd = ta.macd(data['close'], fast=8, slow=17, signal=9)
    data['macd_line'] = macd['MACD_8_17_9']
    data['macd_signal'] = macd['MACDs_8_17_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=25).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35) + (data['lsma_slope'] >= 0) * 1.5
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35) - (data['lsma_slope'] <= 0) * 1.5


    
    # Generate signals
    data['is_buy2'] = ((abs(data['open'] - data['lsma_lower_band'])<0.1) |\
                        (data['open']<data['lsma_lower_band']))&\
                      (data['open'] < data['close']) & \
                      (data['open'].shift(1) > data['close'].shift(1)) & \
                      (((data['close'] - data['low']) + 2) < 5.5)

    data['is_sell2'] = ((abs(data['open'] - data['lsma_upper_band'])<0.1) |\
                        (data['open']>data['lsma_upper_band']))&\
                       (data['open'] > data['close']) & \
                       (data['open'].shift(1) < data['close'].shift(1)) & \
                       (((data['high'] - data['close']) + 2) < 5.5)
    
    # Set take profit and stop loss
    data.loc[data['is_buy2'], 'tp'] = data['close'] + 7
    data.loc[data['is_buy2'], 'sl'] = data['low'] - 2
    data.loc[data['is_sell2'], 'tp'] = data['close'] - 7
    data.loc[data['is_sell2'], 'sl'] = data['high'] + 2

    #set new trailling stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 2
    data.loc[data['is_sell2'], 'be'] = data['close'] - 2

    #condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + 2.7
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - 2.7
    
    return data


