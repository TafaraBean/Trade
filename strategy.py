import pandas as pd
import plotly.graph_objects as go
import pandas_ta as ta

# Create a sample dataframe with some price data
data = {
    'date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05', 
             '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10'],
    'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
    'close': [100, 102, 101, 103, 105, 107, 106, 108, 110, 109]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Convert 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Calculate the Simple Moving Average (SMA) with a period of 3
df['SMA'] = ta.sma(df['close'], length=3)

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Candlestick')])

# Add SMA line
fig.add_trace(go.Scatter(x=df['date'], y=df['SMA'], 
                         mode='lines', name='SMA'))

# Update layout
fig.update_layout(title='Candlestick Chart with SMA',
                  xaxis_title='Date',
                  yaxis_title='Price')

# Show the plot
fig.show()
