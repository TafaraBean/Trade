from datetime import datetime
import MetaTrader5 as mt5
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
from strategy import *
import pytz
# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

# Initialize the trading bot
bot = TradingBot(login=account, password=password, server=server)
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M15   # Change this as needed
start = datetime(2024, 6, 17)
lot= 0.01
#print(bot.get_position_all(symbol=symbol))
#bot.changesltp(symbol=symbol, ticket=18977692, pos_type=mt5.ORDER_TYPE_SELL, sl = 2300, tp=2400,lot=0.01)
#print(bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end))
bot.run(symbol="XAUUSD", timeframe=timeframe, strategy_func=m15_gold_strategy, start=start, lot=lot)