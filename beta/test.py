import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import *

# Load data
data = pd.read_csv('beta/output.csv')
data['time'] = data['time'].astype('datetime64[s]')
data = data.set_index('time')

# Take natural log of data to resolve price scaling issues
data = np.log(data)

# Trendline parameter
lookback = 30

# Function to calculate trendlines for each window
support_trendlines = []
resistance_trendlines = []

# Iterate from the end of the dataset towards the beginning
for i in range(len(data) - 1, lookback - 2, -lookback):
    window_data = data.iloc[i - lookback + 1: i + 1]
    support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
    support_trendlines.append((window_data.index, support_coefs))
    resistance_trendlines.append((window_data.index, resist_coefs))

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close']
)])

# Add support and resistance lines for each window
for dates, support_coefs in support_trendlines:
    if not np.isnan(support_coefs[0]):
        support_start = support_coefs[0] * 0 + support_coefs[1]
        support_end = support_coefs[0] * (lookback - 1) + support_coefs[1]
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[support_start, support_end],
            mode='lines',
            name='Support Line',
            line=dict(color='green')
        ))

for dates, resist_coefs in resistance_trendlines:
    if not np.isnan(resist_coefs[0]):
        resist_start = resist_coefs[0] * 0 + resist_coefs[1]
        resist_end = resist_coefs[0] * (lookback - 1) + resist_coefs[1]
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]],
            y=[resist_start, resist_end],
            mode='lines',
            name='Resistance Line',
            line=dict(color='red')
        ))

# Update layout
fig.update_layout(
    title='Candlestick Chart with Support and Resistance Lines for Each Window',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Show the figure
fig.show()
