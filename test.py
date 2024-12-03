from mt5linux import MetaTrader5 as mt5
from dotenv import load_dotenv
import os
import json
import pandas as pd

# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()
else:
    print("MetaTrader5 package version: ",mt5.__version__)
