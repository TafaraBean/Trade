import MetaTrader5 as mt5
import pandas as pd
import os


account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")


if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()
else:
    print("MetaTrader5 package version: ",mt5.__version__)

# Attempt to login to the trade account
if not mt5.login(login=account, server=server, password=password):
    print(f"Failed to connect to trade account {account}, error code = {mt5.last_error()}")
    quit()
else:
    print("connected to account #{}".format(account))
    
 

print("100.1" < "900")
