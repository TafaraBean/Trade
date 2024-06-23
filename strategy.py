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
    
    macd = ta.macd(data['close'], fast=20, slow=29, signal=9)
    data['macd_line'] = macd['MACD_20_29_9']
    data['macd_signal'] = macd['MACDs_20_29_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=25).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35) + (data['lsma_slope'] >= 0) * 1.5
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35) - (data['lsma_slope'] <= 0) * 1.5


    
    # Generate signals
    data['is_buy2'] = (data['close'].shift(1) < data['lsma'].shift(1)) & (data['close'] > data['lsma'])&\
                        (data['tick_volume']>data['tick_volume'].shift(1))&\
                        (data['support_gradient']>0) & (data['macd_line']>0) & (data['resistance_gradient']>0)
                        
                       

    data['is_sell2'] = (data['close'].shift(1) > data['lsma'].shift(1)) & (data['close'] < data['lsma'])&\
                        (data['tick_volume']>data['tick_volume'].shift(1))&\
                        (data['resistance_gradient']<0) & (data['macd_line']<0) & (data['resistance_gradient']<0)
                        
                
    
    # Set take profit and stop loss
    data.loc[data['is_buy2'], 'tp'] = data['close'] + 3.2
    data.loc[data['is_buy2'], 'sl'] = data['close'] - 3.2
    data.loc[data['is_sell2'], 'tp'] = data['close'] - 3
    data.loc[data['is_sell2'], 'sl'] = data['close'] + 3

    #set new trailling stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 0.9
    data.loc[data['is_sell2'], 'be'] = data['close'] - 0.9

    #condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + 1
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - 1
    

    return data


