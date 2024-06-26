import MetaTrader5 as mt5
import time
import pandas as pd
from typing import Callable

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
    def __init__(self, login: int, password: str, server: str):
        self.account = Account()
        self.login = login
        self.password = password
        self.server = server
        self.positions = {}
        self.initialize_bot()

        self.timeframe_to_interval = {
            mt5.TIMEFRAME_M1: "min",
            mt5.TIMEFRAME_M2: "2min",
            mt5.TIMEFRAME_M3: "3min",
            mt5.TIMEFRAME_M4: "4min",
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_M6: "6min",
            mt5.TIMEFRAME_M10: "10min",
            mt5.TIMEFRAME_M12: "12min",
            mt5.TIMEFRAME_M15: "15min",
            mt5.TIMEFRAME_M20: "20min",
            mt5.TIMEFRAME_M30: "30min",
            mt5.TIMEFRAME_H1: "h",
            mt5.TIMEFRAME_H2: "2h",
            mt5.TIMEFRAME_H3: "3h",
            mt5.TIMEFRAME_H4: "4h",
            mt5.TIMEFRAME_H6: "6h",
            mt5.TIMEFRAME_H8: "8h",
            mt5.TIMEFRAME_H12: "12h",
            mt5.TIMEFRAME_D1: "d",
        }

    def initialize_bot(self) -> None:
        # Initialize the MetaTrader 5 connection
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            quit()
        else:
           print("MetaTrader5 package version: ",mt5.__version__)
        # Attempt to login to the trade account
        if not mt5.login(self.login, password=self.password, server=self.server):
            print(f"Failed to connect to trade account {self.login}, error code = {mt5.last_error()}")
            quit()
        else:
            print("connected to account #{}".format(self.login))
    
    def open_buy_position(self, symbol: str, lot: float, sl: float=0.0, tp: float=0.0) -> dict:
        """
        Open a buy order for a given symbol. by default 0,0 means no sl or tp will be set
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

    def open_sell_position(self, symbol: str, lot: float, sl: float=0.0, tp: float=0.0) -> dict:
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
    
    def positions_total(self) -> int:
        '''Get the number of open positions.'''
        total=mt5.positions_total()
        return total
    
    def orders_total(self) -> int:
        '''Get the number of active orders.'''
        total=mt5.orders_total()
        return total
    
    #this close position function needs to be refined
    def close_position(self, position_id: int, lot: float, symbol: str) -> dict:
        """
        Close the position for a given symbol.
        """
         # create a close request
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

    def get_position(self,ticket: int) -> dict:
        """
        Get the current position for a given symbol.
        """
        order=mt5.positions_get(ticket=ticket)
        trade_position =order[0]
        return trade_position._asdict()
    
    def get_position_all(self,symbol: str) -> pd.DataFrame:
        """Retrives all positions for a specified symbol"""
        positions=mt5.positions_get(symbol=symbol)

        if len(positions) != 0:
            df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
            df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)
            df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            df = pd.DataFrame(positions)
        
        return df

    def cal_profit(self, symbol: str, order_type, lot: float, open_price: float, close_price:float) -> float:

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

        
        if order_type == mt5.ORDER_TYPE_BUY:
            buy_profit=mt5.order_calc_profit(mt5.ORDER_TYPE_BUY,symbol,lot,open_price, close_price)
            
            if buy_profit!=None:            
                return buy_profit
            else:
                print("order_calc_profit(ORDER_TYPE_BUY) failed, error code =",mt5.last_error())
                raise ValueError("Profit value not a number")
        
        elif order_type == mt5.ORDER_TYPE_SELL:
            sell_profit=mt5.order_calc_profit(mt5.ORDER_TYPE_SELL,symbol,lot,open_price, close_price)
            if sell_profit!=None:
                return sell_profit
            else:
                print("order_calc_profit(ORDER_TYPE_SELL) failed, error code =",mt5.last_error())
                raise ValueError("Profit value not a number")
        else:
            raise ValueError("Invalid order type")
                
    def copy_chart_range(self, symbol: str, timeframe, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        "Retrives chart data from specified start date till end date"
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
        ohlc_data = pd.DataFrame(ohlc_data)
        # Convert 'date' column to datetime type
        if len(ohlc_data) != 0:
            ohlc_data['time'] = pd.to_datetime(ohlc_data['time'],unit='s')
        return ohlc_data

    def copy_chart_count(self, symbol: str, timeframe, start: pd.Timestamp, count: int) -> pd.DataFrame:
        "Retrives chart data from specified start date till end date"
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
        ohlc_data = pd.DataFrame(ohlc_data)
        # Convert 'date' column to datetime type
        if len(ohlc_data) != 0:
            ohlc_data['time'] = pd.to_datetime(ohlc_data['time'],unit='s')
        return ohlc_data

    def shutdown(self) -> None:
        mt5.shutdown()
        print("MetaTrader 5 connection closed")

    def get_ticks_range(self,symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # request ticks start date till end date
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        ticks = pd.DataFrame(ticks)
        if len(ticks) != 0:
            ticks['time'] = pd.to_datetime(ticks['time'],unit='s')
        return ticks
       
    def get_ticks_count(self,symbol: str, start: pd.Timestamp, count: int) -> pd.DataFrame:
        # request  ticks from start date with specified count of candles
        ticks = mt5.copy_ticks_from(symbol, start, count, mt5.COPY_TICKS_ALL)
        ticks = pd.DataFrame(ticks)
        if len(ticks) != 0:
            ticks['time'] = pd.to_datetime(ticks['time'],unit='s')
        return ticks
        
    def changesltp(self, ticket: str, symbol: str, sl:float ,tp: float) -> dict:
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
        result = mt5.order_send(request)._asdict()

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("4. order_send failed, retcode={}".format(result.retcode))

        return result

<<<<<<< HEAD
    def chart(self, symbol, timeframe, start, end) -> pd.DataFrame:
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
        df = pd.DataFrame(ohlc_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader 5 connection closed")

    def auto_trendline(self,data):
        data['time2'] = data['time'].astype('datetime64[s]')
        data = data.set_index('time', drop=True)
        print("hourly data:")
        print(data)
        print("==========")

        print("appling auto trendline...")
        data['time2'] = data['time'].astype('datetime64[s]')
        data = data.set_index('time', drop=True)
        print("hourly data:")
        print(data)
        print("==========")

        # Take natural log of data to resolve price scaling issues
        df_log = np.log(data[['high', 'low', 'close']])

        # Trendline parameter
        lookback = 8

        # Initialize columns for trendlines and their gradients
        data['support_trendline'] = np.nan
        data['resistance_trendline'] = np.nan
        data['support_gradient'] = np.nan
        data['resistance_gradient'] = np.nan
        data['hour_lsma'] = ta.linreg(data['close'], length=8)
        data['prev_hour_lsma']=data['hour_lsma'].shift(1)
        data['hour_lsma_slope'] = data['hour_lsma'].diff()
        data['prev_hour_lsma_slope']= data['hour_lsma_slope'].shift(1)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['hour_macd_line'] = macd['MACD_12_26_9']
        data['prev_hour_macd_line']=data['hour_macd_line'].shift(1)


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
            support_value = support_slope * window_data.at[current_index,'low'] + support_intercept
            resist_value = resist_slope * window_data.at[current_index,'high'] + resist_intercept
            data.at[current_index, 'fixed_support_trendline'] = np.exp(support_value)
            data.at[current_index, 'fixed_resistance_trendline'] = np.exp(resist_value)
            # Apply the calculated gradients to each candle in the window
            
            for j in range(lookback):
                idx = i - lookback + j
                support_value = support_slope * j + support_intercept
                resist_value = resist_slope * j + resist_intercept
                data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
                data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
                data.at[data.index[idx], 'support_gradient'] = support_slope
                data.at[data.index[idx], 'resistance_gradient'] = resist_slope
        return data

    def run(self, symbol, timeframe, start, strategy_func, lot):
=======
    def symbol_info_tick(self, symbol: str) -> dict:
        #get the latests tick information for a specified symbol
        symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
        return symbol_info_tick_dict
    
    def run(self, symbol: str, timeframe, start: pd.Timestamp, strategy_func: Callable[[pd.DataFrame],pd.DataFrame], lot: float) -> None:
>>>>>>> 3658e7a25e8d52a967ba1957f874002ff731cfc6
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
            
            
            df = self.copy_chart_range(symbol=symbol, timeframe=timeframe, start=start, end=end)
            hour_data = self.copy_chart_range(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)



            hour_data=self.auto_trendline(hour_data)
            hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline']]

            df['hourly_time']=df['time'].dt.floor('h')

            merged_data = pd.merge(df,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))

            df = strategy_func(merged_data)

            # Check for new trading signals
            latest_signal = df.iloc[-1]

            # Open orders based on the latest signal
            if latest_signal['is_buy2']:
                self.open_buy_position(symbol=symbol, lot=lot, tp=latest_signal['tp'] , sl=latest_signal['sl'])
            elif latest_signal["is_sell2"]:
                self.open_sell_position(symbol=symbol, lot=lot, tp=latest_signal['tp'], sl=latest_signal['sl'])

            #trail any stop losses as needed
            open_positions_df = self.get_position_all(symbol=symbol)

            for index, row in open_positions_df.iterrows():
                #condition to check how far above opwn price a candle should close before sl is adjusted for buy orders
                if(row['type'] == mt5.ORDER_TYPE_BUY and row['price_current'] >= row['price_open'] + 150):                        
                    self.changesltp(ticket=int(row['ticket']), 
                                    symbol=symbol, 
                                    sl=float(row['price_open'] +100), # how much should the stop loss be adjusted above entry
                                    tp=row['tp'])
                    print(f"sl adjusted for position {row['ticket']} ")

                #condition to check how far below opwn price a candle should close before sl is adjusted for sell orders
                elif(row['type'] == mt5.ORDER_TYPE_SELL and row['price_current'] <= row['price_open'] -150):
                    self.changesltp(ticket=int(row['ticket']), 
                                    symbol=symbol, 
                                    sl=float(row['price_open']-100),# how much should the stop loss be adjusted below entry
                                    tp=row['tp'])
                    print(f"sl adjusted for position {row['ticket']} ")
            # Calculate and display performance metrics
            df.to_csv('main.csv', index=False)

