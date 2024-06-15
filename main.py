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

# Mapping of MT5 timeframes to sleep durations in seconds
timeframe_to_interval = {
    mt5.TIMEFRAME_M1: "min",
    mt5.TIMEFRAME_M5: "5min",
    mt5.TIMEFRAME_M10: "10min",
    mt5.TIMEFRAME_M15: "15min",
    mt5.TIMEFRAME_M30: "30min",
    mt5.TIMEFRAME_H1: "H",
    mt5.TIMEFRAME_H4: "4H",
    mt5.TIMEFRAME_D1: "D",
}

bot.run(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, strategy_func=h1_gold_strategy, start=start, end=end)