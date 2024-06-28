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
    data['lsma'] = ta.linreg(data['close'], length=18)
    
    macd = ta.macd(data['close'], fast=20, slow=29, signal=9)
    data['macd_line'] = macd['MACD_20_29_9']
    data['macd_signal'] = macd['MACDs_20_29_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=25).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    data['prev_lsma_slope_1'] = data['lsma_slope'].shift(1)
    data['prev_lsma_slope_2'] = data['lsma_slope'].shift(2)
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35) + (data['lsma_slope'] >= 0) * 1.5
    data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35) - (data['lsma_slope'] <= 0) * 1.5

    #stochastic 
    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['stoch_k'] = stochastic['STOCHk_14_3_3']
    data['stoch_d'] = stochastic['STOCHd_14_3_3']




    
    # Generate signals
    data['is_buy2'] =(data['close'].shift(1) < data['hour_lsma'].shift(1)) & (data['close'] > data['hour_lsma'])&\
                        (data['macd_line']>data['macd_signal']) &(data['lsma_slope']>0.5) & (data['fixed_support_gradient']>0) & (data['fixed_resistance_gradient']>0)& \
                      (data['stoch_k'] > data['stoch_d'])
                        
                       

    data['is_sell2'] = (data['close'].shift(1) > data['hour_lsma'].shift(1)) & (data['close'] < data['hour_lsma'])&\
                         (data['macd_line']<data['macd_signal']) &(data['lsma_slope']<-0.5) & (data['fixed_support_gradient']<0) & (data['fixed_resistance_gradient']<0)& \
                       (data['stoch_k'] < data['stoch_d'])
                
    
    # Set take profit and stop loss
    data.loc[data['is_buy2'], 'tp'] = data['close'] + 400
    data.loc[data['is_buy2'], 'sl'] = data['close'] - 200
    data.loc[data['is_sell2'], 'tp'] = data['close'] - 400
    data.loc[data['is_sell2'], 'sl'] = data['close'] + 200

    #set new trailling stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 100
    data.loc[data['is_sell2'], 'be'] = data['close'] - 100

    #condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + 150
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - 150
    
    return data


