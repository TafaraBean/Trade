import MetaTrader5 as mt5
import time
import pandas as pd
import os
from typing import Callable
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


"""
Example account info
account_info() as dataframe
              property                      value
0                login                   25115284
1           trade_mode                          0
2             leverage                        100
3         limit_orders                        200
4       margin_so_mode                          0
5        trade_allowed                       True
6         trade_expert                       True
7          margin_mode                          2
8      currency_digits                          2
9           fifo_close                      False
10             balance                    99588.3
11              credit                          0
12              profit                     -45.13
13              equity                    99543.2
14              margin                      54.37
15         margin_free                    99488.8
16        margin_level                     183085
17      margin_so_call                         50
18        margin_so_so                         30
19      margin_initial                          0
20  margin_maintenance                          0
21              assets                          0
22         liabilities                          0
23  commission_blocked                          0
24                name                James Smith
25              server            MetaQuotes-Demo
26            currency                        USD
27             company  MetaQuotes Software Corp.
"""
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
    def __init__(self, login: int, password: str, server: str, symbol: str, timeframe, lot: float):
        self.account = Account()
        self.login = login
        self.password = password
        self.server = server
        self.positions = {}
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot = lot
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
    

    def open_buy_position(self, 
                          symbol: str, 
                          conditions_arr: str,
                          lot: float, 
                          sl: float=0.0, 
                          tp: float=0.0,
                          ) -> dict:
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
            "comment": conditions_arr,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # some filling modes are not supported by some brokers
        }
        print(f"conditions arr {conditions_arr}")
        order=mt5.order_send(request)._asdict()

        if order['retcode'] == mt5.TRADE_RETCODE_DONE:
            # Add the order to the positions dictionary
            self.positions[order['order']] = symbol
            print("Long Position Successfully opened")
        else:
            print("Long position failed to open")

        print(order)
        return order

    def open_sell_position(self, 
                           symbol: str,
                           conditions_arr: str, 
                           lot: float, 
                           sl: float=0.0, 
                           tp: float=0.0) -> dict:
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
            "comment": conditions_arr,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # some filing modes are not supported by some brokers
        }
                    
        order=mt5.order_send(request)._asdict()

        if order['retcode'] == mt5.TRADE_RETCODE_DONE:
            # Add the order to the positions dictionary
            self.positions[order['order']] = symbol
        print(order)
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
        ohlc_data = mt5.copy_rates_range(symbol, timeframe, start, end)
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
        
    def changesltp(self, conditions_arr: str, ticket: str, symbol: str, sl:float ,tp: float) -> dict:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(sl),
            "tp": float(tp),
            #"deviation": 20,
            "magic": 234000,
            "comment": conditions_arr,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
            #"ENUM_ORDER_STATE": mt5.ORDER_FILLING_RETURN,
        }
        #// perform the check and display the result 'as is'
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("4. order_send failed for adjusting SL, retcode={}".format(result.retcode))

        return result

    def symbol_info_tick(self, symbol: str) -> dict:
        #get the latests tick information for a specified symbol
        symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
        return symbol_info_tick_dict
    
    def display_chart(df):
        # Create the subplots with 2 rows
        levels = df['sr_levels']
        unique_levels_set = set()

        # Iterate through each row's sr_levels column
        for levels in levels:
            if levels:  # Ensure levels is not None
                unique_levels_set.update(levels)  # Add levels to the set

        # Convert the set to a sorted list for easier reading/processing
        unique_levels_list = sorted(unique_levels_set)

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
        
        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['lsma3_smooth'], 
                                mode='lines', 
                                name='LSMA3_smooth'
                                ), row=1, col=1)
        

        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['lsma2_smooth'], 
                                mode='lines', 
                                name='LSMA2_smooth'
                                ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['long_smooth'], 
                                mode='lines', 
                                name='long_smooth'
                                ), row=1, col=1)
        
        
        
        



        # Add LMSA Band line to the first subplot
        fig.add_trace(go.Scatter(x=df['time'], y=df['lsma_high'], 
                                mode='lines', name='LSMA_high'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'], y=df['lsma_low'], 
                                mode='lines', name='LSMA_low'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['fixed_support_trendline_15'], 
                                mode='lines', 
                                name='support'
                                ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'], 
                                y=df['fixed_resistance_trendline_15'], 
                                mode='lines', 
                                name='resistance'
                                ), row=1, col=1)
        
        # Plot levels as horizontal lines on the first subplot
        for level in unique_levels_list:  # Iterate through each unique level
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=[level] * len(df),  # Create a horizontal line at the level
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name=f'Level {level:.2f}'
            ), row=1, col=1)

        
        
        
        #Add Bollinger Bands to the first subplot
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['bb_upper'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Bollinger Upper'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['bb_middle'],
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Bollinger Middle'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['bb_lower'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Bollinger Lower'
        ), row=1, col=1)


        # Add MACD Line to the second subplot
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Span_B'],
            name='Span_B',
            line=dict(color='purple')
        ), row=2, col=1)

        # Add MACDs Line to the second subplot
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['Span_A'],
            name='Span_A',
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
    

    def run(self, strategy_func: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]) -> None:
        while True:
            start = pd.Timestamp.now() - pd.Timedelta(days=3) #always use 1 week worth of data to ensure there is enough candle sticks for the  dataframe
            # Calculate the time to sleep until the next interval based on the timeframe
            conversion = self.timeframe_to_interval.get(self.timeframe, 3600) #conversion is used to keep a consistant timeframe thorugh all trade executions
            current_time = pd.Timestamp.now() + pd.Timedelta(hours=1)
            end =  pd.Timestamp.now() + pd.Timedelta(days=1)
            next_interval = current_time.ceil(conversion)
            
           
            print(f"current time: {current_time}")
            print(f"\nSleeping for {(next_interval - current_time)} until the next interval.")
            #time.sleep(10)
            time.sleep((next_interval - current_time).total_seconds())

            
            df = strategy_func(start,end)
            TradingBot.display_chart(df)
            df.to_csv('csv/main.csv', index=False)

            # Check for new trading signals
            
            latest_signal = df.iloc[-1]
            
            # Open orders based on the latest signal
            if latest_signal['is_buy2']:
                order = self.open_buy_position(symbol=self.symbol, conditions_arr=latest_signal['conditions_arr'],lot=self.lot, tp=latest_signal['tp'] , sl=latest_signal['sl'])


            elif latest_signal["is_sell2"]:
                order = self.open_sell_position(symbol=self.symbol, conditions_arr=latest_signal['conditions_arr'], lot=self.lot, tp=latest_signal['tp'], sl=latest_signal['sl'])


            # Track rows to keep
            # get the list of all positions to see if they need to be changed"
            running_positions=self.get_position_all(symbol=self.symbol)
            
            #Note that the variable row['comment'] stores the be_condition
            for index, row in running_positions.iterrows():
                """
                in the conditions array, the following is stored at these indexes
                index 0: be_condition_increment
                index 1: be_increment
                index 2: be_condition
                """
                conditions_arr = list(map(float, row['comment'].split(','))) 
                be_condition_inc = conditions_arr[0]
                be_inc  = conditions_arr[1]
                be_condition = conditions_arr[2]
                
                
                updated_sl = row['sl']  + be_inc  if row['type'] == mt5.ORDER_TYPE_BUY else row['sl']  - be_inc
                updated_be_condition = be_condition + be_condition_inc if row['type'] == mt5.ORDER_TYPE_BUY else be_condition - be_condition_inc
                
                updated_conditions_arr = ",".join(map(str, [be_condition_inc, be_inc, updated_be_condition ]))

                if row['type'] == mt5.ORDER_TYPE_BUY and row['price_current'] >= be_condition:
                    result = self.changesltp(ticket=int(row['ticket']), be_condition=updated_conditions_arr, symbol=self.symbol, sl=float(updated_sl), tp=row['tp'])
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"sl adjusted for position {row['ticket']} ")

                
                # Condition to check how far below open price a candle should close before sl is adjusted for sell orders
                elif row['type'] == mt5.ORDER_TYPE_SELL and row['price_current'] <= be_condition:
                    result = self.changesltp(ticket=int(row['ticket']), be_condition=updated_conditions_arr, symbol=self.symbol, sl=float(updated_sl),  tp=row['tp'])
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"sl adjusted for position {row['ticket']} ")                            
                        # Update the DataFrame
