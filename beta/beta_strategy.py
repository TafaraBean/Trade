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



        data.loc[data['is_buy2'], 'tp'] = data['low'] + 9
        data.loc[data['is_buy2'], 'sl'] = data['low'] - 3
        data.loc[data['is_sell2'], 'tp'] = data['high'] - 9
        data.loc[data['is_sell2'], 'sl'] = data['high'] + 3

        return data

def m15_gold_strategy(data):
        data['ema_short'] = ta.ema(data['close'], length=12)
        data['ema_long'] = ta.ema(data['close'], length=26)
        data['lsma'] = ta.linreg(data['close'], length=25)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['macd_line'] = macd['MACD_12_26_9']
        data['lsma_stddev'] = data['close'].rolling(window=25).std()
        data['lsma_upper_band'] = data['lsma'] + (data['lsma_stddev'] * 1.35)
        data['lsma_lower_band'] = data['lsma'] - (data['lsma_stddev'] * 1.35)

        data['is_buy2'] = (data['low'] < data['lsma_lower_band']) & (data['open'] < data['close']) & \
                                (data['open'].shift(1) > data['close'].shift(1)) & (data['macd_line'] < 0)
        data['is_sell2'] = (data['high'] > data['lsma_upper_band']) & (data['open'] > data['close']) & \
                                (data['open'].shift(1) < data['close'].shift(1)) & (data['macd_line'] > 0)

        data.loc[data['is_buy2'], 'tp'] = data['low'] + 7
        data.loc[data['is_buy2'], 'sl'] = data['low'] - 3
        data.loc[data['is_sell2'], 'tp'] = data['high'] - 7
        data.loc[data['is_sell2'], 'sl'] = data['high'] + 3

        return data

def calc_prof(trade_data):
        # Sample trade data for demonstration
        # This would typically come from your trading results


        # Convert to DataFrame
        df_trades = pd.DataFrame(trade_data)

        # Calculate total gross profit
        total_gross_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()

        # Calculate total gross loss
        total_gross_loss = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())

        # Calculate number of profitable trades
        num_profitable_trades = df_trades[df_trades['profit'] > 0].shape[0]

        # Calculate number of losing trades
        num_losing_trades = df_trades[df_trades['profit'] < 0].shape[0]

        # Calculate net profit
        net_profit = total_gross_profit - total_gross_loss

        # Calculate profit factor
        if total_gross_loss > 0:
                profit_factor = total_gross_profit / total_gross_loss
        else:
                profit_factor = float('inf')  # To handle division by zero if there are no losses

        # Calculate percentage profitability
        total_trades = df_trades.shape[0]
        if total_trades > 0:
                percentage_profitability = (num_profitable_trades / total_trades) * 100
        else:
                percentage_profitability = 0

        # Print results
        print(f"Total Gross Profit: ${total_gross_profit}")
        print(f"Total Gross Loss: ${total_gross_loss}")
        print(f"Net Profit: ${net_profit}")
        print(f"Profit Factor: {profit_factor}")
        print(f"Percentage Profitability: {percentage_profitability}%")
