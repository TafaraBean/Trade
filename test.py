import pandas as pd
from trading_bot import TradingBot
from dotenv import load_dotenv
import os
import MetaTrader5 as mt5
# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

# Initialize the trading bot
bot = TradingBot(login=account, password=password, server=server)

start = pd.Timestamp("2024-07-26")
end = pd.Timestamp.now() + pd.Timedelta(days=1)

df = bot.copy_chart_range(symbol="EURUSD.Z",end=end, start=start,timeframe=mt5.TIMEFRAME_H1)

print(df)
print(start)
print(end)