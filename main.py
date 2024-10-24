import MetaTrader5 as mt5
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
from strategy import *


# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M15  # Change this as needed

# Initialize the trading bot
bot = TradingBot(login=account, 
                 password=password, 
                 server=server, 
                 symbol = symbol,
                 timeframe = timeframe)



lot= 0.01

#bot.run(timeframe=timeframe, strategy_func=apply_strategy, lot=lot)

