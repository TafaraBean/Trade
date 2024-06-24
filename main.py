from datetime import datetime
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
timeframe = mt5.TIMEFRAME_M15   # Change this as needed
start = pd.Timestamp("2024-06-18")
lot= 0.02
bot.run(symbol="XAUUSD", timeframe=timeframe, strategy_func=m15_gold_strategy, start=start, lot=lot)
