import pandas as pd
from trading_bot import TradingBot
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
account = int(os.environ.get("ACCOUNT"))
password = os.environ.get("PASSWORD")
server = os.environ.get("SERVER")

# Initialize the trading bot
bot = TradingBot(login=account, password=password, server=server)

# Load DataFrame
df_trades =  pd.read_csv('csv/executed_trades_df.csv')
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
df_trades['profit'] = df_trades['profit'].astype(float)

# Define a function to get market price at a specific time
def get_market_price(symbol, time):
    # This is a placeholder function. Replace it with the actual implementation.
    # Example return value
    return bot.get_ticks_count(symbol=symbol, start=time,count=1).iloc[0]['bid']

    #return 100.0  # You should replace this with actual market data retrieval logic

# Generate 5-minute intervals for the analysis
start_time = df_trades['entry_time'].min()
end_time = df_trades['entry_time'].max()  # End time of the analysis period
time_intervals = pd.date_range(start=start_time, end=end_time, freq='10min')

# Create an empty DataFrame to store the equity calculations
equity_df = pd.DataFrame(time_intervals, columns=['time'])
equity_df['equity'] = 0.0

# Function to calculate running trade profit
def calculate_running_trade_profit(trade, current_time):
    # Get the market price at the current time
    market_price = get_market_price("EURUSD.Z", current_time)
    
    # Placeholder for profit calculation logic
    # You need to implement how to calculate profit based on entry price and market price
    entry_price = 100  # Example placeholder
    profit = bot.profit_loss("EURUSD.Z", trade['order_type'], trade['lot_size'],trade['close'], market_price)
    
    if isinstance(profit,float):
        return profit
    else:
        return 0
# Calculate equity for each time interval
for index, row in equity_df.iterrows():
 
    if  index  % 100 == 0:
        print(index)
    current_time = row['time']
    
    # Calculate the profit for closed trades
    closed_trades_profit = df_trades[
        (df_trades['exit_time'] <= current_time) & (df_trades['exit_time'].notna())
    ]['profit'].sum()
    
    # Calculate the profit for running trades
    running_trades = df_trades[
        (df_trades['entry_time'] <= current_time) & (df_trades['exit_time'] >= current_time)
    ]

    if not running_trades.empty:
        running_trades_profit = running_trades.apply(lambda trade: calculate_running_trade_profit(trade, current_time), axis=1).sum()
    else:
        running_trades_profit = 0.0
    # Calculate total equity
    equity_df.at[index, 'equity'] = 1000+ closed_trades_profit + running_trades_profit

print(equity_df)
equity_df.to_csv('csv/equity.csv', index=False)