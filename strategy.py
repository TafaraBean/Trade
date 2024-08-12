import pandas as pd
import pandas_ta as ta
import numpy as np
import MetaTrader5 as mt5
import talib


def h1_gold_strategy(data):
        data['ema_short'] = ta.ema(data['close'], length=12)
        data['ema_long'] = ta.ema(data['close'], length=26)
        data['lsma'] = ta.linreg(data['close'], length=100)
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
    data['lsma'] = ta.linreg(data['close'], length=7)
    
    macd = ta.macd(data['close'], fast=12, slow=24, signal=9)
    data['macd_line'] = macd['MACD_12_24_9']
    data['macd_signal'] = macd['MACDs_12_24_9']
    
    data['lsma_stddev'] = data['close'].rolling(window=7).std()
    
    # Identify the trend
    data['lsma_slope'] = data['lsma'].diff()
    data['prev_lsma_slope_1'] = data['lsma_slope'].shift(1)
    data['prev_lsma_slope_2'] = data['lsma_slope'].shift(2)
    
    # Adjust LSMA bands based on trend
    data['lsma_upper_band'] = data['lsma'] + data['lsma_stddev']
    data['lsma_lower_band'] = data['lsma'] - data['lsma_stddev']

    # Calculate stochastic oscillator
    stochastic = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['stoch_k'] = stochastic['STOCHk_14_3_3']
    data['stoch_d'] = stochastic['STOCHd_14_3_3']

    data['is_doji'] = talib.CDLDOJI(data['open'],data['high'],data['low'],data['close'])
    data['is_doji_star'] = talib.CDLDOJISTAR(data['open'],data['high'],data['low'],data['close'])
    data['3whitesoldiers'] = talib.CDL3WHITESOLDIERS(data['open'],data['high'],data['low'],data['close'])
    data['3blackcrows'] = talib.CDL3BLACKCROWS(data['open'],data['high'],data['low'],data['close'])
    data['engulfing'] = talib.CDLENGULFING(data['open'],data['high'],data['low'],data['close'])





    pip_size = 0.0001

    # Set TP and SL in terms of pips
    tp_pips = 100 * pip_size
    sl_pips = 40 * pip_size
    be_pips = 5 * pip_size
    data['ticket'] = np.nan

    
    # Generate signals
    data['is_buy2'] = (
        
          (data['close']>data['prev_fixed_resistance_trendline'])&
          (data['close'].shift(1)<data['prev_fixed_resistance_trendline'].shift(1))&
          (data['Span_A']>data['Span_B'])
          

         


        
    )

    data['is_sell2'] = (
        (data['close']<data['prev_fixed_support_trendline'])&
        (data['close'].shift(1)>data['prev_fixed_support_trendline'].shift(1))&
        (data['Span_A']<data['Span_B'])
          
    )
    

    
    data.loc[data['is_buy2'], 'signal'] = mt5.ORDER_TYPE_BUY
    data.loc[data['is_sell2'], 'signal'] = mt5.ORDER_TYPE_SELL 
    data.loc[data['is_buy2'], 'tp'] = data['close'] + tp_pips
    data.loc[data['is_buy2'], 'sl'] = data['close'] - sl_pips
    data.loc[data['is_sell2'], 'tp'] = data['close'] - tp_pips
    data.loc[data['is_sell2'], 'sl'] = data['close'] + sl_pips

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 2 * pip_size 
    data.loc[data['is_sell2'], 'be'] = data['close'] - 2 * pip_size

    # Condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + be_pips
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - be_pips

    return data