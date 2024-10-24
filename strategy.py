import pandas as pd
import pandas_ta as ta
import numpy as np
import MetaTrader5 as mt5
import talib
from main import bot
from analysis import auto_trendline_15, auto_trendline, auto_trendline_4H

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



def apply_strategy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = bot.copy_chart_range(symbol=bot.symbol, timeframe=bot.timeframe, start=start, end=end)
    data=auto_trendline_15(data)

    hour_data = bot.copy_chart_range(symbol=bot.symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)

    hour_data=auto_trendline(hour_data)

    hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline','prev_fixed_support_trendline','prev_fixed_resistance_trendline','prev_fixed_resistance_gradient','prev_fixed_support_gradient','ema_24','prev_stochk','prev_stochd','prev_hour_macd_signal','prev_psar','prev_psar_direction','prev_nadaraya_watson','prev_nadaraya_watson_trend','nadaraya_upper_envelope','nadaraya_lower_envelope','wma_50','prev_supertrend_dir','HSpan_A','HSpan_B','nadaraya_watson','prev_nadaraya_lower_band','prev_nadaraya_upper_band','prev_HSpan_A','prev_HSpan_B','support_trendline','resistance_trendline','support','resistance','is_buy','is_sell']]

    hour_data.to_csv("csv/hour_data.csv",index=False)

    data['hourly_time']=data['time'].dt.floor('h')

    merged_data = pd.merge(data,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))

    four_hour_data = bot.copy_chart_range(symbol=bot.symbol, timeframe=mt5.TIMEFRAME_H4, start=start, end=end)
    four_hour_data = auto_trendline_4H(four_hour_data)
    H4_data = four_hour_data[['time4h','4H_ema_50','prev_4fixed_support_gradient','prev_4fixed_support_trendline','prev_4fixed_resistance_gradient','prev_4fixed_resistance_trendline','prev_4nadaraya_watson_trend','4H_hour_lsma','4lsma_upper_band','4lsma_lower_band','4HSpan_A','4HSpan_B']]

    merged_data['4_hour_time']=merged_data['time2'].dt.floor('4h')


    merged_data2 = pd.merge(merged_data,H4_data, left_on='4_hour_time', right_on='time4h', suffixes=('', '_4h'))

    return m15_gold_strategy(merged_data2)


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

    # Calculate the candlestick patterns
    data['is_doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
    data['is_doji_star'] = talib.CDLDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['3whitesoldiers'] = talib.CDL3WHITESOLDIERS(data['open'], data['high'], data['low'], data['close'])
    data['3blackcrows'] = talib.CDL3BLACKCROWS(data['open'], data['high'], data['low'], data['close'])
    data['engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
    data['inverted_hammer'] = talib.CDLINVERTEDHAMMER(data['open'], data['high'], data['low'], data['close'])
    data['two_crows'] = talib.CDL2CROWS(data['open'], data['high'], data['low'], data['close'])
    data['three_inside'] = talib.CDL3INSIDE(data['open'], data['high'], data['low'], data['close'])
    data['three_line_strike'] = talib.CDL3LINESTRIKE(data['open'], data['high'], data['low'], data['close'])
    data['three_outside'] = talib.CDL3OUTSIDE(data['open'], data['high'], data['low'], data['close'])
    data['three_stars_in_south'] = talib.CDL3STARSINSOUTH(data['open'], data['high'], data['low'], data['close'])
    data['abandoned_baby'] = talib.CDLABANDONEDBABY(data['open'], data['high'], data['low'], data['close'])
    data['advance_block'] = talib.CDLADVANCEBLOCK(data['open'], data['high'], data['low'], data['close'])
    data['belt_hold'] = talib.CDLBELTHOLD(data['open'], data['high'], data['low'], data['close'])
    data['breakaway'] = talib.CDLBREAKAWAY(data['open'], data['high'], data['low'], data['close'])
    data['closing_marubozu'] = talib.CDLCLOSINGMARUBOZU(data['open'], data['high'], data['low'], data['close'])
    data['conceal_baby_swallow'] = talib.CDLCONCEALBABYSWALL(data['open'], data['high'], data['low'], data['close'])
    data['counterattack'] = talib.CDLCOUNTERATTACK(data['open'], data['high'], data['low'], data['close'])
    data['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(data['open'], data['high'], data['low'], data['close'])
    data['doji'] = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
    data['doji_star'] = talib.CDLDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(data['open'], data['high'], data['low'], data['close'])
    data['engulfing'] = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])
    data['evening_doji_star'] = talib.CDLEVENINGDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['evening_star'] = talib.CDLEVENINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['gap_side_side_white'] = talib.CDLGAPSIDESIDEWHITE(data['open'], data['high'], data['low'], data['close'])
    data['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(data['open'], data['high'], data['low'], data['close'])
    data['hammer'] = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
    data['hanging_man'] = talib.CDLHANGINGMAN(data['open'], data['high'], data['low'], data['close'])
    data['harami'] = talib.CDLHARAMI(data['open'], data['high'], data['low'], data['close'])
    data['harami_cross'] = talib.CDLHARAMICROSS(data['open'], data['high'], data['low'], data['close'])
    data['high_wave'] = talib.CDLHIGHWAVE(data['open'], data['high'], data['low'], data['close'])
    data['hikkake'] = talib.CDLHIKKAKE(data['open'], data['high'], data['low'], data['close'])
    data['hikkake_mod'] = talib.CDLHIKKAKEMOD(data['open'], data['high'], data['low'], data['close'])
    data['homing_pigeon'] = talib.CDLHOMINGPIGEON(data['open'], data['high'], data['low'], data['close'])
    data['identical_three_crows'] = talib.CDLIDENTICAL3CROWS(data['open'], data['high'], data['low'], data['close'])
    data['in_neck'] = talib.CDLINNECK(data['open'], data['high'], data['low'], data['close'])
    data['inverted_hammer'] = talib.CDLINVERTEDHAMMER(data['open'], data['high'], data['low'], data['close'])
    data['kicking'] = talib.CDLKICKING(data['open'], data['high'], data['low'], data['close'])
    data['kicking_by_length'] = talib.CDLKICKINGBYLENGTH(data['open'], data['high'], data['low'], data['close'])
    data['ladder_bottom'] = talib.CDLLADDERBOTTOM(data['open'], data['high'], data['low'], data['close'])
    data['long_legged_doji'] = talib.CDLLONGLEGGEDDOJI(data['open'], data['high'], data['low'], data['close'])
    data['long_line'] = talib.CDLLONGLINE(data['open'], data['high'], data['low'], data['close'])
    data['matching_low'] = talib.CDLMATCHINGLOW(data['open'], data['high'], data['low'], data['close'])
    data['mat_hold'] = talib.CDLMATHOLD(data['open'], data['high'], data['low'], data['close'])
    data['morning_doji_star'] = talib.CDLMORNINGDOJISTAR(data['open'], data['high'], data['low'], data['close'])
    data['morning_star'] = talib.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['on_neck'] = talib.CDLONNECK(data['open'], data['high'], data['low'], data['close'])
    data['piercing'] = talib.CDLPIERCING(data['open'], data['high'], data['low'], data['close'])
    data['rickshaw_man'] = talib.CDLRICKSHAWMAN(data['open'], data['high'], data['low'], data['close'])
    data['rise_fall_3_methods'] = talib.CDLRISEFALL3METHODS(data['open'], data['high'], data['low'], data['close'])
    data['separating_lines'] = talib.CDLSEPARATINGLINES(data['open'], data['high'], data['low'], data['close'])
    data['shooting_star'] = talib.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['short_line'] = talib.CDLSHORTLINE(data['open'], data['high'], data['low'], data['close'])
    data['spinning_top'] = talib.CDLSPINNINGTOP(data['open'], data['high'], data['low'], data['close'])
    data['stalled_pattern'] = talib.CDLSTALLEDPATTERN(data['open'], data['high'], data['low'], data['close'])
    data['stick_sandwich'] = talib.CDLSTICKSANDWICH(data['open'], data['high'], data['low'], data['close'])
    data['takuri'] = talib.CDLTAKURI(data['open'], data['high'], data['low'], data['close'])
    data['tasuki_gap'] = talib.CDLTASUKIGAP(data['open'], data['high'], data['low'], data['close'])
    data['thrusting'] = talib.CDLTHRUSTING(data['open'], data['high'], data['low'], data['close'])
    data['tristar'] = talib.CDLTRISTAR(data['open'], data['high'], data['low'], data['close'])
    data['unique_3_river'] = talib.CDLUNIQUE3RIVER(data['open'], data['high'], data['low'], data['close'])
    data['upside_gap_two_crows'] = talib.CDLUPSIDEGAP2CROWS(data['open'], data['high'], data['low'], data['close'])
    data['xside_gap_3_methods'] = talib.CDLXSIDEGAP3METHODS(data['open'], data['high'], data['low'], data['close'])

    data['hanging_man'].to_csv("csv/hanging",index=False)
    data['inverted_hammer'].to_csv("csv/inverted_hammer",index=False)







    pip_size = 1

    # Set TP and SL in terms of pips
    tp_pips = 100 * pip_size
    sl_pips = 30 * pip_size
    be_pips = 35 * pip_size
    data['ticket'] = np.nan

    # Generate signals
    data['is_buy2'] = (
        ((data['lsma']>data['ema_50'])&
        (data['lsma'].shift(1)<data['ema_50'].shift(1)))
    )

    data['is_sell2'] = (
       ((data['lsma']<data['ema_50'])&
        (data['lsma'].shift(1)>data['ema_50'].shift(1)))
    )

    

    
    data.loc[data['is_buy2'], 'signal'] = mt5.ORDER_TYPE_BUY
    data.loc[data['is_sell2'], 'signal'] = mt5.ORDER_TYPE_SELL 
    data.loc[data['is_buy2'], 'tp'] = data['close'] + tp_pips
    data.loc[data['is_buy2'], 'sl'] = data['close'] - sl_pips
    data.loc[data['is_sell2'], 'tp'] = data['close'] - tp_pips
    data.loc[data['is_sell2'], 'sl'] = data['close'] + sl_pips

    # Set new trailing stop loss
    data.loc[data['is_buy2'], 'be'] = data['close'] + 30 * pip_size 
    data.loc[data['is_sell2'], 'be'] = data['close'] - 30 * pip_size

    # Condition for setting new trailing stop
    data.loc[data['is_buy2'], 'be_condition'] = data['close'] + be_pips
    data.loc[data['is_sell2'], 'be_condition'] = data['close'] - be_pips

    return data


