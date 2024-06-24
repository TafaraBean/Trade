import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import fit_trendlines_high_low  # Ensure this function is correctly imported from your analysis module

# Load data
data = pd.read_csv('beta/output.csv')
data['time'] = data['time'].astype('datetime64[s]')
data = data.set_index('time', drop=False)

# Take natural log of data to resolve price scaling issues
df_log = np.log(data[['high', 'low', 'close']])

# Trendline parameter
lookback = 15

# Initialize columns for trendlines and their gradients
data['support_trendline'] = np.nan
data['resistance_trendline'] = np.nan
data['support_gradient'] = np.nan
data['resistance_gradient'] = np.nan

# Lists to hold the trendline segments
support_trendline_x = []
support_trendline_y = []
resistance_trendline_x = []
resistance_trendline_y = []

# Focus only on the last 15 candles
if len(df_log) >= lookback:
    window_data = df_log.iloc[-lookback:]
    support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
    
    # Get the gradient from the first candle in the window (the 15th last candle overall)
    initial_support_gradient = support_coefs[0]
    initial_resistance_gradient = resist_coefs[0]

    # Apply this gradient to the current 15-candle window
    for j in range(lookback):
        idx = len(df_log) - lookback + j
        support_value = initial_support_gradient * j + support_coefs[1]
        resist_value = initial_resistance_gradient * j + resist_coefs[1]
        data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
        data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
        data.at[data.index[idx], 'support_gradient'] = initial_support_gradient
        data.at[data.index[idx], 'resistance_gradient'] = initial_resistance_gradient
        
        # Append to trendline segments
        support_trendline_x.append(data.index[idx])
        support_trendline_y.append(np.exp(support_value))
        resistance_trendline_x.append(data.index[idx])
        resistance_trendline_y.append(np.exp(resist_value))
    
    # Append None to break the line
    support_trendline_x.append(None)
    support_trendline_y.append(None)
    resistance_trendline_x.append(None)
    resistance_trendline_y.append(None)

# Print the data
print(data)
data.to_csv('beta/test.csv', index=False)

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close']
)])

# Add support and resistance lines
fig.add_trace(go.Scatter(
    x=support_trendline_x,
    y=support_trendline_y,
    mode='lines',
    name='Support Line',
    line=dict(color='green'),
    connectgaps=False,
    line_shape='linear'
))

fig.add_trace(go.Scatter(
    x=resistance_trendline_x,
    y=resistance_trendline_y,
    mode='lines',
    name='Resistance Line',
    line=dict(color='red'),
    connectgaps=False,
    line_shape='linear'
))

# Update layout
fig.update_layout(
    title='Candlestick Chart with Support and Resistance Lines',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Show the figure
fig.show()
