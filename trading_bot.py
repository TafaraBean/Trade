import MetaTrader5 as mt5
import time
import pandas as pd
import os
from typing import Callable
from analysis import *
from utils import *
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
        self.positions_file_path = 'csv/positions.csv'
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
        if not mt5.login(login=self.login, server=self.server, password=self.password):
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

    def profit_loss(self, 
                    symbol: str, 
                    order_type:int , 
                    lot: float, 
                    open_price: float, 
                    close_price:float) -> float:

        #fetch symbol data        
        symbol_info=mt5.symbol_info(symbol)
        if symbol_info is None:
            print(symbol,"not found, skipped")
            return 0
        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(symbol,True):
                print("symbol_select({}}) failed, skipped",symbol)
                return 0

        
        if order_type == mt5.ORDER_TYPE_BUY:
            return mt5.order_calc_profit(mt5.ORDER_TYPE_BUY,
                                         symbol,
                                         lot,
                                         open_price,
                                         close_price)
     

            #    print("order_calc_profit(ORDER_TYPE_BUY) failed, error code =",mt5.last_error())
             #   raise ValueError("Profit value not a number")
        
        elif order_type == mt5.ORDER_TYPE_SELL:
            return mt5.order_calc_profit(mt5.ORDER_TYPE_SELL,
                                              symbol,
                                              lot,
                                              open_price, 
                                              close_price)
            
                 
        
                #print("order_calc_profit(ORDER_TYPE_SELL) failed, error code =",mt5.last_error())
                #raise ValueError("Profit value not a number")
        else:
            raise ValueError("Invalid order type")

    def copy_chart_range(self, symbol: str, timeframe, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        "Retrives chart data from specified start date till end date"
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start.to_pydatetime(), end.to_pydatetime())
        ohlc_data = pd.DataFrame(ohlc_data)
        # Convert 'date' column to datetime type
        if len(ohlc_data) != 0:
            ohlc_data['time'] = pd.to_datetime(ohlc_data['time'],unit='s')
        return ohlc_data

    def copy_chart_count(self, symbol: str, timeframe, start: pd.Timestamp, count: int) -> pd.DataFrame:
        "Retrives chart data from specified start date till end date"
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start.to_pydatetime(), count)
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
        ticks = mt5.copy_ticks_range(symbol, start.to_pydatetime(), end.to_pydatetime(), mt5.COPY_TICKS_ALL)
        ticks = pd.DataFrame(ticks)
        if len(ticks) != 0:
            ticks['time'] = pd.to_datetime(ticks['time'],unit='s')
        return ticks
       
    def get_ticks_count(self,symbol: str, start: pd.Timestamp, count: int) -> pd.DataFrame:
        # request  ticks from start date with specified count of candles
        ticks = mt5.copy_ticks_from(symbol, start.to_pydatetime(), count, mt5.COPY_TICKS_ALL)
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
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("4. order_send failed, retcode={}".format(result.retcode))

        return result

    def symbol_info_tick(self, symbol: str) -> dict:
        #get the latests tick information for a specified symbol
        symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
        return symbol_info_tick_dict
    
    def display_chart(df):
            # Create the subplots with 2 rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[0.8, 0.2],
                            subplot_titles=('Candlestick Chart', 'MACD Line'))

        # Add candlestick chart to the first subplot
        fig.add_trace(go.Candlestick(x=df['time'],
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='Candlestick'), row=1, col=1)

        # Add buy signals (up arrows) to the first subplot
        fig.add_trace(go.Scatter(
            x=df[df['is_buy2'] == True]['time'],
            y=df[df['is_buy2'] == True]['low'] * 0.999,
            mode='markers',
            marker=dict(symbol='arrow-up', color='green', size=10),
            name='Buy Signal'
        ), row=1, col=1)

        # Add sell signals (down arrows) to the first subplot
        fig.add_trace(go.Scatter(
            x=df[df['is_sell2'] == True]['time'],
            y=df[df['is_sell2'] == True]['high'] * 1.001,
            mode='markers',
            marker=dict(symbol='arrow-down', color='red', size=10),
            name='Sell Signal'
        ), row=1, col=1)



        
        
    

        

        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['ema_50'], 
                                mode='lines', 
                                name='ema_50'
                                ), row=1, col=1)
        
        



        # Add LMSA Band line to the first subplot
        fig.add_trace(go.Scatter(x=df['time'], y=df['lsma'], 
                                mode='lines', name='LMSA'), row=1, col=1)

        # Add MACD Line to the second subplot
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['macd_line'],
            name='MACD Line',
            line=dict(color='purple')
        ), row=2, col=1)

        # Add MACDs Line to the second subplot
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['macd_signal'],
            name='MACD Signal',
            line=dict(color='blue')
        ), row=2, col=1)


        # Update layout
        fig.update_layout(title='XAUUSD',
                        #xaxis_title='Date',
                        #yaxis_title='Price',
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark"
                        )

        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), #hide weekends
            ]
        )
        
        fig.show()

    
    def run(self, symbol: str, timeframe, strategy_func: Callable[[pd.DataFrame],pd.DataFrame], lot: float) -> None:
        while True:
            start = pd.Timestamp.now() + pd.Timedelta(hours=1) - pd.Timedelta(days=4)
            # Calculate the time to sleep until the next interval based on the timeframe
            # Get current time
            conversion = self.timeframe_to_interval.get(timeframe, 3600)
            current_time = pd.Timestamp.now() + pd.Timedelta(hours=1)
            next_interval = current_time.ceil(conversion)
            
            # Calculate the difference in seconds
            time_difference = (next_interval - current_time).total_seconds()
            end = pd.to_datetime(current_time).floor(conversion)
            print(f"current time: {current_time}")
            print(f"\nSleeping for {time_difference / 60.0} miniutes until the next interval.")
            time.sleep(time_difference)

            # Fetch the market data and apply the trading strategy
            
            
            df = self.copy_chart_range(symbol=symbol, timeframe=timeframe, start=start, end=end)
            df=auto_trendline_15(df)
            
            hour_data = self.copy_chart_range(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, start=start, end=end)
            hour_data= auto_trendline(hour_data)

            hourly_data = hour_data[['time2','prev_hour_lsma_slope','prev_hour_macd_line','hour_lsma','fixed_support_gradient','fixed_resistance_gradient','prev_hour_lsma','fixed_support_trendline','fixed_resistance_trendline','prev_fixed_support_trendline','prev_fixed_resistance_trendline','prev_fixed_resistance_gradient','prev_fixed_support_gradient','ema_50','ema_24','prev_stochk','prev_stochd','prev_hour_macd_signal','prev_psar','prev_psar_direction','prev_nadaraya_watson','prev_nadaraya_watson_trend','nadaraya_upper_envelope','nadaraya_lower_envelope','wma_50','prev_supertrend_dir','HSpan_A','HSpan_B','nadaraya_watson','prev_nadaraya_lower_band','prev_nadaraya_upper_band','prev_HSpan_A','prev_HSpan_B','support_trendline','resistance_trendline','lsma']]

            df['hourly_time']=df['time'].dt.floor('h')

            merged_data = pd.merge(df,hourly_data, left_on='hourly_time', right_on='time2', suffixes=('_15m', '_hourly'))

            df = strategy_func(merged_data)

            display_chart(df)
            

            # Check for new trading signals
            
            latest_signal = df.iloc[-1]
            
            # Open orders based on the latest signal
            if latest_signal['is_buy2']:
                order = self.open_buy_position(symbol=symbol, lot=lot, tp=latest_signal['tp'] , sl=latest_signal['sl'])
                df.at[df.index[-1], 'ticket'] = order['order']
                df.at[df.index[-1], 'type'] = mt5.ORDER_TYPE_BUY

            elif latest_signal["is_sell2"]:
                order = self.open_sell_position(symbol=symbol, lot=lot, tp=latest_signal['tp'], sl=latest_signal['sl'])
                df.at[df.index[-1], 'ticket'] = order['order']
                df.at[df.index[-1], 'type'] = mt5.ORDER_TYPE_SELL

            required_columns = ['ticket', 'sl', 'tp', 'be', 'be_condition']
            try:
                # Try to open the file in read mode
                with open(self.positions_file_path, 'r') as file:
                    headers_df = pd.DataFrame(columns=required_columns)
                    headers_df.to_csv(self.positions_file_path, mode='a', header=not pd.read_csv(self.positions_file_path).empty, index=False)
        
            except FileNotFoundError:
                # If the file does not exist, create it
                with open(self.positions_file_path, 'w') as file:
                    headers_df = pd.DataFrame(columns=required_columns)
                    headers_df.to_csv(self.positions_file_path, header=True, index=False)



            with open(self.positions_file_path, 'a') as f:
                if latest_signal['is_buy2'] or latest_signal['is_sell2']:
                # Convert Series to DataFrame before writing to CSV
                    last_row = df.tail(1)
                    last_row.to_csv(self.positions_file_path, header=True, index=False, columns=required_columns)


            df.to_csv('main.csv', index=False)
            
            #trail any stop losses as needed
            open_positions_df = self.get_position_all(symbol=symbol)
            csv_positions_df = pd.read_csv('csv/positions.csv')

            if 'ticket' in open_positions_df.columns:
                valid_tickets = set(open_positions_df['ticket'])

            
        
            

            # Track rows to keep
            

            for index, row in open_positions_df.iterrows():
                specific_row_index = csv_positions_df.index[csv_positions_df['ticket'] == row['ticket']]
                
                if specific_row_index.empty:
                    add_missing_position(order=row,
                                        file_path=self.positions_file_path)
                    # Reload positions_df again to get the updated content
                    csv_positions_df = pd.read_csv('csv/positions.csv')
                    
                    # Find the new index of the added position
                    specific_row_index = csv_positions_df.index[csv_positions_df['ticket'] == row['ticket']]
    

                if not specific_row_index.empty:

                    # Extract the specific row index
                    idx = specific_row_index[0]
                    
                    # Extract the specific values
                    be_condition = csv_positions_df.at[idx, 'be_condition']
                    be = csv_positions_df.at[idx, 'be']
                    
                    # Condition to check how far above open price a candle should close before sl is adjusted for buy orders
                    if row['type'] == mt5.ORDER_TYPE_BUY and row['price_current'] >= be_condition:
                        result = self.changesltp(ticket=int(row['ticket']), 
                                                symbol=symbol, 
                                                sl=float(be),  # how much should the stop loss be adjusted above entry
                                                tp=row['tp'])
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"sl adjusted for position {row['ticket']} ")
                            # Update the DataFrame
                            csv_positions_df.at[idx, 'be_condition'] += 5 * 0.0001
                            csv_positions_df.at[idx, 'be'] += 4 * 0.0001
                    
                    # Condition to check how far below open price a candle should close before sl is adjusted for sell orders
                    elif row['type'] == mt5.ORDER_TYPE_SELL and row['price_current'] <= be_condition:
                        result = self.changesltp(ticket=int(row['ticket']), 
                                                symbol=symbol, 
                                                sl=float(be),  # how much should the stop loss be adjusted below entry
                                                tp=row['tp'])
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"sl adjusted for position {row['ticket']} ")                            
                            # Update the DataFrame
                            csv_positions_df.at[idx, 'be_condition']  -= 5 * 0.0001
                            csv_positions_df.at[idx, 'be'] -= 4 * 0.0001
                else:
                    print(f"No specific row found for ticket {row['ticket']}")

            # Save the updated DataFrame back to the CSV file
            # Save the updated DataFrame back to the CSV file
            if 'ticket' in open_positions_df.columns:
                rows_to_keep = csv_positions_df[csv_positions_df['ticket'].isin(valid_tickets)]
                rows_to_keep.to_csv('csv/positions.csv', index=False)