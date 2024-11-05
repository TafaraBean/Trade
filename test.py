import MetaTrader5 as mt5
from dotenv import load_dotenv
import os



# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()
else:
    print("MetaTrader5 package version: ",mt5.__version__)

# Attempt to login to the trade account
if not mt5.login(login=account, server=server, password=password):
    print(f"Failed to connect to trade account {account}, error code = {mt5.last_error()}")
    quit()
else:
    print("connected to account #{}".format(account))
    

request = {
"action": mt5.TRADE_ACTION_DEAL,
"symbol": "XAUUSD",
"volume": 0.01,
"type": mt5.ORDER_TYPE_BUY,
"price": mt5.symbol_info_tick("XAUUSD").ask,
"sl": 0.0,
"tp": 0.0,
"deviation": 20, #tolorance for order filling when market moves after sending order
"magic": 234000, #each order can be uniquely identified per EA
"comment": "conditions_arr",
"type_time": mt5.ORDER_TIME_GTC,
"type_filling": mt5.ORDER_FILLING_FOK, # some filling modes are not supported by some brokers
}


symbol = "XAUUSD"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(symbol, "not found, can not call order_check()")
    mt5.shutdown()
    quit()
 
# if the symbol is unavailable in MarketWatch, add it
if not symbol_info.visible:
    print(symbol, "is not visible, trying to switch on")
    if not mt5.symbol_select(symbol,True):
        print("symbol_select({}}) failed, exit",symbol)
        mt5.shutdown()
        quit()
 

#
print(mt5.symbol_info_tick("USDJPY"))
print(mt5.order_send(request)._asdict())