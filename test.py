from main import bot
import pandas as pd
from analysis import auto_trendline_15
import time

df = (bot.copy_chart_count_pos(bot.symbol, bot.timeframe,1, 400))

#start_time = time.time()
print(df)

#end_time = time.time()
# Calculate the duration
#execution_time = end_time - start_time
#print(f"Execution time: {execution_time:.5f} seconds")

