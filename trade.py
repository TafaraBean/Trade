import time
from trading_bot import TradingBot
from dotenv import load_dotenv
import os

load_dotenv()
account=int(os.environ.get("ACCOUNT"))
password=os.environ.get("PASSWORD")
server=os.environ.get("SERVER")


bot = TradingBot( account=account, password=password, server=server)




