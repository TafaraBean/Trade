import MetaTrader5 as mt5
import time
import pandas as pd

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
        self.initialize_bot()

        self.timeframe_to_interval = {
            mt5.TIMEFRAME_M1: "min",
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_M10: "10min",
            mt5.TIMEFRAME_M15: "15min",
            mt5.TIMEFRAME_M30: "30min",
            mt5.TIMEFRAME_H1: "H",
            mt5.TIMEFRAME_H4: "4H",
            mt5.TIMEFRAME_D1: "D",
        }

    def initialize_bot(self):
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

    def close_position(self, position_id, lot, symbol) -> dict:
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

    def cal_profit(self, symbol, order_type, lot, open_price, close_price) -> float:

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
                return None
        
        elif order_type == mt5.ORDER_TYPE_SELL:
            sell_profit=mt5.order_calc_profit(mt5.ORDER_TYPE_SELL,symbol,lot,open_price, close_price)
            if sell_profit!=None:
                return sell_profit
            else:
                print("order_calc_profit(ORDER_TYPE_SELL) failed, error code =",mt5.last_error())
                return None
        else:
            print("Invalid order type")
            return None
                
    def chart(self, symbol, timeframe, start, end) -> pd.DataFrame:
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
        ohlc_data = pd.DataFrame(ohlc_data)
        # Convert 'date' column to datetime type
        ohlc_data['time'] = pd.to_datetime(ohlc_data['time'],unit='s')
        return ohlc_data

    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader 5 connection closed")

    def get_ticks(self,symbol, start, end) -> pd.DataFrame:
        # request AUDUSD ticks within 11.01.2020 - 11.01.2020
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        ticks = pd.DataFrame(ticks)
        ticks['time'] = pd.to_datetime(ticks['time'],unit='s')
        return ticks
    
    
    def get_ticks_count(self,symbol, start, count) -> pd.DataFrame:
        # request AUDUSD ticks within 11.01.2020 - 11.01.2020
        ticks = mt5.copy_ticks_from(symbol, start, count, mt5.COPY_TICKS_ALL)
        ticks = pd.DataFrame(ticks)
        ticks['time'] = pd.to_datetime(ticks['time'],unit='s')
        return ticks
        
        
    def symbol_info_tick(self, symbol) -> dict:
        symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
        return symbol_info_tick_dict

    def run(self, symbol, timeframe, start, strategy_func):
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
            df = strategy_func(df)

            # Check for new trading signals
            latest_signal = df.iloc[-1]

            # Open orders based on the latest signal
            if latest_signal["is_buy2"]:
                self.open_buy_order(symbol=symbol, lot=0.01, tp=latest_signal['tp'] , sl=latest_signal['sl'])
            elif latest_signal["is_sell2"]:
                self.open_sell_order(symbol=symbol, lot=0.01, tp=latest_signal['tp'], sl=latest_signal['sl'])


            # Calculate and display performance metrics
            df.to_csv('output.csv', index=False)

