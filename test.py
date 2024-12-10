from main import bot

df = (bot.copy_chart_count_pos(bot.symbol, bot.timeframe,1, 400))
position = bot.get_position_all(bot.symbol)
#start_time = time.time()


#end_time = time.time()
# Calculate the duration
#execution_time = end_time - start_time
#print(f"Execution time: {execution_time:.5f} seconds")

