import pandas_ta as ta

def apply_strategy(data):
        data['ema_short'] = ta.ema(data['close'], length=12)
        data['ema_long'] = ta.ema(data['close'], length=26)
        data['lsma'] = ta.linreg(data['close'], length=25)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['macd_line'] = macd['MACD_12_26_9']
        data['lsma_stddev'] = data['close'].rolling(window=25).std()
        data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35)
        data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35)

        data['is_buy2'] = (data['low'] < data['lsma_lower_band']) & (data['open'] < data['close']) & \
                          (data['open'].shift(1) > data['close'].shift(1)) & (data['macd_line'] > 0)
        data['is_sell2'] = (data['high'] > data['lsma_upper_band']) & (data['open'] > data['close']) & \
                           (data['open'].shift(1) < data['close'].shift(1)) & (data['macd_line'] < 0)
        return data
