import MetaTrader5 as mt5
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
from strategy import *
import threading


# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

# Initialize the trading bots
bot1 = TradingBot(login=account, password=password, server=server)
bot2 = TradingBot(login=account, password=password, server=server)

symbol1 = "BTCUSD"
symbol2 = "ETHUSD"
timeframe1 = mt5.TIMEFRAME_M15  # Change this as needed
timeframe2 = mt5.TIMEFRAME_H1   # Change this as needed

lot1 = 0.01
lot2 = 0.02

# Function to run a trading bot in a thread
def run_bot(bot, symbol, timeframe, strategy_func, lot):
    print(f"Running {symbol} bot")
    bot.run(symbol=symbol, timeframe=timeframe, strategy_func=strategy_func, lot=lot)

# Create threads for each bot
thread1 = threading.Thread(target=run_bot, args=(bot1, symbol1, timeframe1, m15_gold_strategy, lot1))
thread2 = threading.Thread(target=run_bot, args=(bot2, symbol2, timeframe2, m15_gold_strategy, lot2))

# Start the threads
thread1.start()
thread2.start()

# Optionally, join the threads to wait for their completion
# (This is not needed if they run indefinitely)
# thread1.join()
# thread2.join()
