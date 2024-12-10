import MetaTrader5 as mt5
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
from strategy import apply_strategy


# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M1 # Change this as needed
lot= 0.01
# Initialize the trading bot object 
bot = TradingBot(login=account, 
                 password=password, 
                 server=server, 
                 symbol = symbol,
                 timeframe = timeframe,
                 lot = lot)

if __name__ == "__main__":
    
    bot.run(strategy_func=apply_strategy, close_opposing_trades=True)

