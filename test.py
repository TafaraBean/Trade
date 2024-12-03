# import the package
from mt5linux import MetaTrader5
# connecto to the server
mt5 = MetaTrader5(
    host = 'localhost',
     port = 18812
) 
# use as you learned from: https://www.mql5.com/en/docs/integration/python_metatrader5/
mt5.initialize()
mt5.terminal_info()
mt5.copy_rates_from_pos('VALE3',mt5.TIMEFRAME_M1,0,1000)
# ...
# don't forget to shutdown
mt5.shutdown()