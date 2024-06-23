import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import *
#THIS ONE ONLY RETRIEVES THE LAST 30 DAYS!
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

# Get the data for the last 30 candles
window_data = df_log.iloc[-lookback:]

# Fit trendlines to the high and low values of the last 30 candles
support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])

# Calculate trendline values and gradients for each point in the window
support_trendline_x = []
support_trendline_y = []
resistance_trendline_x = []
resistance_trendline_y = []

for j in range(lookback):
    idx = len(df_log) - lookback + j
    support_value = support_coefs[0] * j + support_coefs[1]
    resist_value = resist_coefs[0] * j + resist_coefs[1]
    data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
    data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
    data.at[data.index[idx], 'support_gradient'] = support_coefs[0]
    data.at[data.index[idx], 'resistance_gradient'] = resist_coefs[0]

    # Append to trendline segments
    support_trendline_x.append(data.index[idx])
    support_trendline_y.append(np.exp(support_value))
    resistance_trendline_x.append(data.index[idx])
    resistance_trendline_y.append(np.exp(resist_value))

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