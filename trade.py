import time
from trading_bot import TradingBot


bot = TradingBot( account=account, password=password, server=server)


print(bot.get_position(ticket=18610280))
