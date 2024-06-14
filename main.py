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

def main():
    while True:
        # Calculate the time to sleep until the next interval based on the timeframe
        # Get current time
        conversion = timeframe_to_interval.get(timeframe, 3600)
        current_time = pd.Timestamp.now()
        next_interval = current_time.ceil(conversion)
        
        # Calculate the difference in seconds
        time_difference = (next_interval - current_time).total_seconds()
        
        print(f"Sleeping for {time_difference / 60.0} miniutes until the next interval.")
        time.sleep(time_difference)

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
