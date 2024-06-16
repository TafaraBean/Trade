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
        return data



