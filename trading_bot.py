import MetaTrader5 as mt5
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import *
from datetime import datetime

class Account:
    def __init__(self):
        self._info = None

    def update_info(self):
        account_info = mt5.account_info()._asdict()
        self._info = account_info

    def __getattr__(self, attr):
        if attr == "login":
            self.update_info()  # Update account info every time login is accessed
            return self._info.get("login", None)
        elif self._info is None:
            self.update_info()  # Update account info if not already fetched
        if attr in self._info:
            return self._info[attr]
        raise AttributeError(f"'Account' object has no attribute '{attr}'")
    
class TradingBot:   
    def __init__(self, login, password, server):
        self.account = Account()
        self.login = login
        self.password = password
        self.server = server
        self.positions = {}
        self.initialize_api()

        self.timeframe_to_interval = {
            mt5.TIMEFRAME_M1: "min",
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_M10: "10min",
            mt5.TIMEFRAME_M15: "15min",
            mt5.TIMEFRAME_M30: "30min",
            mt5.TIMEFRAME_H1: "h",
            mt5.TIMEFRAME_H4: "4h",
            mt5.TIMEFRAME_D1: "D",
        }

    def initialize_api(self):
        # Initialize the MetaTrader 5 connection
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
        else:
           print("MetaTrader5 package version: ",mt5.__version__)
        # Attempt to login to the trade account
        if not mt5.login(self.login, password=self.password, server=self.server):
            print(f"Failed to connect to trade account {self.login}, error code = {mt5.last_error()}")
        else:
            print("connected to account #{}".format(self.login))
    
    def open_buy_order(self, symbol, lot, sl=0.0, tp=0.0) -> dict:
        """
        Open a buy order for a given symbol.
        """
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20, #tolorance for order filling when market moves after sending order
            "magic": 234000, #each order can be uniquely identified per EA
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # some filling modes are not supported by some brokers
        }
            
        order=mt5.order_send(request)._asdict()

        if order['retcode'] == mt5.TRADE_RETCODE_DONE:
            # Add the order to the positions dictionary
            self.positions[order['order']] = symbol
        
        return order

    def open_sell_order(self, symbol, lot, sl=0.0, tp=0.0) -> dict:
        """
        Open a sell order for a given symbol.
        """
        # Logic to open a sell order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # some filing modes are not supported by some brokers
        }
                    
        order=mt5.order_send(request)._asdict()
        if order['retcode'] == mt5.TRADE_RETCODE_DONE:
            # Add the order to the positions dictionary
            self.positions[order['order']] = symbol
        return order

    def close_position(self, position_id, lot, symbol) -> pd.DataFrame:
        """
        Close the position for a given symbol.
        """
         # create a close request

        order = self.get_position(ticket=position_id)
        trade_type = mt5.order['type']
        request={
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "position": position_id,
            "price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        # send a trading request
        result = mt5.order_send(request)._asdict()
        return result

    def get_position(self,ticket) -> dict:
        """
        Get the current position for a given symbol.
        """
        order=mt5.positions_get(ticket=ticket)
        trade_position =order[0]
        return trade_position._asdict()
    
    def get_position_all(self,symbol) -> pd.DataFrame:
        positions=mt5.positions_get(symbol=symbol)

        if len(positions) != 0:
            df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
            df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)
            df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            df = pd.DataFrame(positions)
        
        return df

    def cal_profit(self, symbol, order_type, lot, distance, tp=0, sl=0):
        # get account currency
        account_currency=self.account.currency

        #fetch symbol data        
        symbol_info=mt5.symbol_info(symbol)
        if symbol_info is None:
            print(symbol,"not found, skipped")
            return None
        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(symbol,True):
                print("symbol_select({}}) failed, skipped",symbol)
                return None
   
        point=mt5.symbol_info(symbol).point
        symbol_tick=mt5.symbol_info_tick(symbol)
        ask=symbol_tick.ask
        bid=symbol_tick.bid

        
        if order_type == mt5.ORDER_TYPE_BUY:
            buy_profit=mt5.order_calc_profit(mt5.ORDER_TYPE_BUY,symbol,lot,ask,ask+distance*point)
            
            if buy_profit!=None:
                #print("   buy {} {} lot: profit on {} points => {} {}".format(symbol,lot,distance,buy_profit,account_currency))
                return buy_profit
            else:
                print("order_calc_profit(ORDER_TYPE_BUY) failed, error code =",mt5.last_error())
        
        elif order_type == mt5.ORDER_TYPE_SELL:
            sell_profit=mt5.order_calc_profit(mt5.ORDER_TYPE_SELL,symbol,lot,bid,bid-distance*point)
            if sell_profit!=None:
                #print("   sell {} {} lots: profit on {} points => {} {}".format(symbol,lot,distance,sell_profit,account_currency))
                
                return sell_profit
            else:
                print("order_calc_profit(ORDER_TYPE_SELL) failed, error code =",mt5.last_error())
        else:
            print("Invalid order type")
            return None

    def changesltp(self, ticket,symbol ,sl,tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
            #"deviation": 20,
            "magic": 234000,
            #"comment": "python script modify",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
            #"ENUM_ORDER_STATE": mt5.ORDER_FILLING_RETURN,
        }
        #// perform the check and display the result 'as is'
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("4. order_send failed, retcode={}".format(result.retcode))

        return result

    def chart(self, symbol, timeframe, start, end) -> pd.DataFrame:
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
        df = pd.DataFrame(ohlc_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader 5 connection closed")

    def auto_trendline(self,data):
        print("applying auto trendline...")
        data['time2'] = data['time'].astype('datetime64[s]')
        data = data.set_index('time', drop=True)

        # Take natural log of data to resolve price scaling issues
        df_log = np.log(data[['high', 'low', 'close']])

        # Trendline parameter
        lookback = 8

        # Initialize columns for trendlines and their gradients
        data['support_trendline'] = np.nan
        data['resistance_trendline'] = np.nan
        data['support_gradient'] = np.nan
        data['resistance_gradient'] = np.nan

        # Iterate over the dataset in overlapping windows of 15 candles
        for i in range(lookback, len(df_log) + 1):
            current_index = df_log.index[i-1]
            window_data = df_log.iloc[i - lookback:i]
            support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
            
            # Extract slope and intercept
            support_slope, support_intercept = support_coefs
            resist_slope, resist_intercept = resist_coefs
            data.at[current_index, 'fixed_resistance_gradient'] = resist_slope
            data.at[current_index, 'fixed_support_gradient'] = support_slope
            # Apply the calculated gradients to each candle in the window
            
            for j in range(lookback):
                idx = i - lookback + j
                support_value = support_slope * j + support_intercept
                resist_value = resist_slope * j + resist_intercept
                data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
                data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
                data.at[data.index[idx], 'support_gradient'] = support_slope
                data.at[data.index[idx], 'resistance_gradient'] = resist_slope

        # Create a candlestick chart
        return data

    def run(self, symbol, timeframe, start, strategy_func, lot):
        while True:
            # Calculate the time to sleep until the next interval based on the timeframe
            # Get current time
            conversion = self.timeframe_to_interval.get(timeframe, 3600)
            current_time = pd.Timestamp.now() + pd.Timedelta(hours=1)
            next_interval = current_time.ceil(conversion)
            
            # Calculate the difference in seconds
            time_difference = (next_interval - current_time).total_seconds()
            end = pd.to_datetime(current_time).floor(conversion)
            print(f"Sleeping for {time_difference / 60.0} miniutes until the next interval.")
            time.sleep(time_difference)

            # Fetch the market data and apply the trading strategy
            
            
            df = self.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)
            hour_data = self.chart(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)



            hour_data=self.auto_trendline(hour_data)
            hourly_data = hour_data[['time2', 'fixed_support_gradient', 'fixed_resistance_gradient']]

            df['hourly_time']=df['time'].dt.floor('h')

            merged_data = pd.merge(df,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))

            df = strategy_func(merged_data)

            # Check for new trading signals
            latest_signal = df.iloc[-1]

            # Open orders based on the latest signal
            if latest_signal['is_buy2']:
                self.open_buy_order(symbol=symbol, lot=lot, tp=latest_signal['tp'] , sl=latest_signal['sl'])
            elif latest_signal["is_sell2"]:
                self.open_sell_order(symbol=symbol, lot=lot, tp=latest_signal['tp'], sl=latest_signal['sl'])

            #trail any stop losses as needed
            open_positions_df = self.get_position_all(symbol=symbol)

            for index, row in open_positions_df.iterrows():
                #condition to check how far above opwn price a candle should close before sl is adjusted for buy orders
                if(row['type'] == mt5.ORDER_TYPE_BUY and row['price_current'] >= row['price_open'] +1.2):                        
                    self.changesltp(ticket=int(row['ticket']), 
                                    symbol=symbol, 
                                    sl=float(row['price_open'] +1), # how much should the stop loss be adjusted above entry
                                    tp=row['tp'])
                    print(f"sl adjusted for position {row['ticket']} ")

                #condition to check how far below opwn price a candle should close before sl is adjusted for sell orders
                elif(row['type'] == mt5.ORDER_TYPE_SELL and row['price_current'] <= row['price_open'] -1.2):
                    self.changesltp(ticket=int(row['ticket']), 
                                    symbol=symbol, 
                                    sl=float(row['price_open']-1),# how much should the stop loss be adjusted below entry
                                    tp=row['tp'])
                    print(f"sl adjusted for position {row['ticket']} ")
            # Calculate and display performance metrics
            df.to_csv('main.csv', index=False)

