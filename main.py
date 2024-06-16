from datetime import datetime, timedelta
import time
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

# Initialize the trading bot
bot = TradingBot(login=account, password=password, server=server)
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_H1  # Change this as needed
start = datetime(2024,5,1)
end = datetime.now()



bot.run(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, strategy_func=h1_gold_strategy, start=start, end=end)