from datetime import datetime, timedelta
import time
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
timezone = pytz.timezone("UTC")
timeframe = mt5.TIMEFRAME_M5   # Change this as needed
start = datetime(2024, 5, 10)


#print(bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end))
bot.run(symbol="XAUUSD", timeframe=timeframe, strategy_func=h1_gold_strategy, start=start)