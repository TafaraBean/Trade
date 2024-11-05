import MetaTrader5 as mt5
from dotenv import load_dotenv
import os
import json
import pandas as pd

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
 

#order_data = mt5.order_send(request)._asdict()


#print(json.dumps(order_data, indent=4))
positions=mt5.positions_get(symbol=symbol)

if len(positions) != 0: 
    df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
    df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
else:
    df = pd.DataFrame(positions)
    
print(df)