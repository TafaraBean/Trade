import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import *

# Load data
data = pd.read_csv('beta/output.csv')
data['time'] = data['time'].astype('datetime64[s]')
data = data.set_index('time')

# Take natural log of data to resolve price scaling issues
df_log = np.log(data[['high', 'low', 'close']])

# Trendline parameter
lookback = 30

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

# Iterate from the end of the dataset towards the beginning
for i in range(len(df_log) - 1, lookback - 2, -1):
    window_data = df_log.iloc[i - lookback + 1: i + 1]
    support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])
    
    # Calculate trendline values and gradients for each point in the window
    for j in range(lookback):
        idx = i - lookback + 1 + j
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
    
    # Append None to break the line
    support_trendline_x.append(None)
    support_trendline_y.append(None)
    resistance_trendline_x.append(None)
    resistance_trendline_y.append(None)

print(data)
data.to_csv('beta/test.csv',index=False)
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
