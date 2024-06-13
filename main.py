from datetime import datetime, timedelta
import time
import MetaTrader5 as mt5
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
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

# Mapping of MT5 timeframes to sleep durations in seconds
timeframe_to_seconds = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_D1: 86400,
}

def main():
    while True:
        # Calculate the time to sleep until the next interval based on the timeframe
        now = datetime.now()
        interval_seconds = timeframe_to_seconds.get(timeframe, 3600)  # Default to 1 hour if timeframe is not found
        next_interval = (now + timedelta(seconds=interval_seconds)).replace(minute=0,second=0, microsecond=0)
        time_to_sleep = (next_interval - now).total_seconds()
        
        print(f"Sleeping for {time_to_sleep} seconds until the next interval.")
        time.sleep(time_to_sleep)
        
        # Define the time range for fetching data
        end = datetime.now()
        start = end - timedelta(days=30)  # Fetch the last 30 days of data

        # Fetch the market data and apply the trading strategy
        data = bot.chart(symbol=symbol, timeframe=timeframe, start=start, end=end)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = apply_strategy(df)

        # Check for new trading signals
        latest_signal = df.iloc[-1]

        # Open orders based on the latest signal
        if latest_signal["is_buy2"]:
            bot.open_buy_order(symbol=symbol, lot=0.01, tp=latest_signal['low'] + 9, sl=latest_signal['low'] - 3)
        elif latest_signal["is_sell2"]:
            bot.open_sell_order(symbol=symbol, lot=0.01, tp=latest_signal['high'] - 9, sl=latest_signal['high'] + 3)

        # Calculate and display performance metrics
        df.to_csv('output.csv', index=False)


if __name__ == "__main__":
    main()
