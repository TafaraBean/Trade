import MetaTrader5 as mt5
import pandas as pd

class TradingBot:
    def __init__(self, account, password, server):
        self.account = account
        self.password = password
        self.server = server
        self.positions = {}
        self.account_info = {}
        self.initialize_api()


    def initialize_api(self):
        # Initialize the MetaTrader 5 connection
        if not mt5.initialize():
            raise ConnectionError(f"initialize() failed, error code = {mt5.last_error()}")
        else:
           print("MetaTrader5 package version: ",mt5.__version__)
        # Attempt to login to the trade account
        if not mt5.login(self.account, password=self.password, server=self.server):
            raise ConnectionError(f"Failed to connect to trade account {self.account}, error code = {mt5.last_error()}")
        else:
            print("connected to account #{}".format(self.account))
        self.account_info = mt5.account_info()._asdict()


    def open_buy_order(self, symbol, volume, sl=0.0, tp=0.0):
        """
        Open a buy order for a given symbol.
        """
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
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

    def open_sell_order(self, symbol, volume, sl=0.0, tp=0.0):
        """
        Open a sell order for a given symbol.
        """
        # Logic to open a sell order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
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

    def close_position(self, position_id, lot, symbol):
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
        result=mt5.order_send(request)

    def get_position(self,ticket) -> tuple:
        """
        Get the current position for a given symbol.
        """
        order=mt5.positions_get(ticket=ticket)
        trade_position =order[0]
        return trade_position._asdict()


    def get_positions(self):
        return self.positions
   
   
    def get_account_balance(self):
        return  self.account_info.get('balance', 0)
    
    def get_account_info(self) -> dict:
        # Retrieve account information
        return self.account_info












    def execute_strategy(self):
        """
        Execute the trading strategy.

        This method should be overridden by subclasses to implement specific trading strategies.
        """
        pass

    def get_login(self):
        return self.account_info.get('login', 'No login information available')

    def get_trade_mode(self):
        return self.account_info.get('trade_mode', 'No trade mode information available')

    def get_leverage(self):
        return self.account_info.get('leverage', 'No leverage information available')

    def get_limit_orders(self):
        return self.account_info.get('limit_orders', 'No limit orders information available')

    def get_margin_so_mode(self):
        return self.account_info.get('margin_so_mode', 'No margin SO mode information available')

    def get_trade_allowed(self):
        return self.account_info.get('trade_allowed', 'No trade allowed information available')

    def get_trade_expert(self):
        return self.account_info.get('trade_expert', 'No trade expert information available')

    def get_margin_mode(self):
        return self.account_info.get('margin_mode', 'No margin mode information available')

    def get_currency_digits(self):
        return self.account_info.get('currency_digits', 'No currency digits information available')

    def get_fifo_close(self):
        return self.account_info.get('fifo_close', 'No FIFO close information available')

    def get_credit(self):
        return self.account_info.get('credit', 0)

    def get_profit(self):
        return self.account_info.get('profit', 0)

    def get_equity(self):
        return self.account_info.get('equity', 0)

    def get_margin(self):
        return self.account_info.get('margin', 0)

    def get_margin_free(self):
        return self.account_info.get('margin_free', 0)

    def get_margin_level(self):
        return self.account_info.get('margin_level', 0)

    def get_margin_so_call(self):
        return self.account_info.get('margin_so_call', 0)

    def get_margin_so_so(self):
        return self.account_info.get('margin_so_so', 0)

    def get_margin_initial(self):
        return self.account_info.get('margin_initial', 0)

    def get_margin_maintenance(self):
        return self.account_info.get('margin_maintenance', 0)

    def get_assets(self):
        return self.account_info.get('assets', 0)

    def get_liabilities(self):
        return self.account_info.get('liabilities', 0)

    def get_commission_blocked(self):
        return self.account_info.get('commission_blocked', 0)

    def get_name(self):
        return self.account_info.get('name', 'No name information available')

    def get_server(self):
        return self.account_info.get('server', 'No server information available')

    def get_currency(self):
        return self.account_info.get('currency', 'No currency information available')

    def get_company(self):
        return self.account_info.get('company', 'No company information available')

    def shutdown(self):
        mt5.shutdown()
        print("MetaTrader 5 connection closed")
